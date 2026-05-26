"""Ollama HTTP client with streaming timing and mock mode for offline dev."""

from __future__ import annotations

import json
import platform
import re
import time
from collections.abc import Iterator
from typing import Any

import httpx

from local_slm.config import settings
from local_slm.models import GenerationResult

# Synthetic profiles when MOCK_OLLAMA=true — relative speed tiers, not real hardware.
_MOCK_PROFILES: dict[str, dict[str, float]] = {
    "llama3.2:1b": {"ttft_ms": 180, "tps": 95, "quality_bias": 0.55},
    "llama3.2:3b": {"ttft_ms": 320, "tps": 52, "quality_bias": 0.72},
    "qwen2.5:3b": {"ttft_ms": 350, "tps": 48, "quality_bias": 0.78},
    "phi3:mini": {"ttft_ms": 400, "tps": 44, "quality_bias": 0.75},
}


def _mock_profile(model: str) -> dict[str, float]:
    if model in _MOCK_PROFILES:
        return _MOCK_PROFILES[model]
    # Fuzzy match on tag suffix
    base = model.split(":")[0]
    for key, profile in _MOCK_PROFILES.items():
        if key.startswith(base):
            return profile
    return {"ttft_ms": 400, "tps": 40, "quality_bias": 0.65}


def _mock_response(model: str, prompt: str, max_tokens: int) -> str:
    profile = _mock_profile(model)
    bias = profile["quality_bias"]
    lower = prompt.lower()

    if "17 * 23" in prompt or "17*23" in prompt:
        return "391" if bias >= 0.6 else "394"
    if "120 miles" in lower and "2 hours" in lower:
        return "60" if bias >= 0.55 else "58"
    if "json" in lower and "status" in lower:
        return '{"status": "ok", "count": 42}' if bias >= 0.65 else '{"status": "OK"}'
    if "summarize" in lower or "privacy" in lower:
        return (
            "Local inference keeps data on-device for privacy and avoids API fees, "
            "at the cost of hardware spend and sometimes higher latency."
            if bias >= 0.6
            else "LLMs can run locally."
        )
    if "bullet" in lower:
        return (
            "- Cold-start latency sets first-impression wait time in chat.\n"
            "- Users abandon flows when the first token takes too long."
            if bias >= 0.62
            else "Cold start is important."
        )
    if "benchmark" in lower:
        return (
            "1. Install Ollama and pull models.\n"
            "2. Warm up with throwaway generations.\n"
            "3. Fix prompt length and max_tokens.\n"
            "4. Record TTFT, total latency, and tokens/sec.\n"
            "5. Compare quality on the same task set."
        )

    words = re.findall(r"\w+", prompt)
    snippet = " ".join(words[:12]) or "your request"
    return (
        f"[mock:{model}] Offline response for: {snippet}. "
        f"Running locally avoids sending data to third-party APIs."
    )[: max(max_tokens * 4, 80)]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        mock: bool | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.mock = settings.mock_ollama if mock is None else mock
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        if self.mock:
            return {"status": "mock", "models": list(_MOCK_PROFILES.keys())}
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            return resp.json()

    def list_models(self) -> list[str]:
        data = self.health()
        if self.mock:
            return list(_MOCK_PROFILES.keys())
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        stream: bool = True,
    ) -> GenerationResult:
        max_tokens = max_tokens or settings.benchmark_max_tokens

        if self.mock:
            return self._mock_generate(model, prompt, max_tokens)

        if stream:
            return self._generate_stream(model, prompt, max_tokens, temperature)
        return self._generate_blocking(model, prompt, max_tokens, temperature)

    def _mock_generate(self, model: str, prompt: str, max_tokens: int) -> GenerationResult:
        profile = _mock_profile(model)
        response = _mock_response(model, prompt, max_tokens)
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = min(_estimate_tokens(response), max_tokens)
        ttft = profile["ttft_ms"]
        gen_ms = (completion_tokens / profile["tps"]) * 1000
        total_ms = ttft + gen_ms
        return GenerationResult(
            model=model,
            prompt=prompt,
            response=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            time_to_first_token_ms=ttft,
            total_latency_ms=total_ms,
            tokens_per_second=completion_tokens / (gen_ms / 1000) if gen_ms else 0.0,
        )

    def _generate_blocking(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        start = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        total_ms = (time.perf_counter() - start) * 1000

        response = data.get("response", "")
        prompt_tokens = data.get("prompt_eval_count") or _estimate_tokens(prompt)
        completion_tokens = data.get("eval_count") or _estimate_tokens(response)
        eval_ms = data.get("eval_duration", 0) / 1_000_000
        tps = completion_tokens / (eval_ms / 1000) if eval_ms else 0.0

        return GenerationResult(
            model=model,
            prompt=prompt,
            response=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            time_to_first_token_ms=None,
            total_latency_ms=total_ms,
            tokens_per_second=tps,
        )

    def _generate_stream(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        start = time.perf_counter()
        ttft_ms: float | None = None
        chunks: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("response"):
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start) * 1000
                        chunks.append(data["response"])
                    if data.get("prompt_eval_count"):
                        prompt_tokens = data["prompt_eval_count"]
                    if data.get("eval_count"):
                        completion_tokens = data["eval_count"]
                    if data.get("done"):
                        break

        total_ms = (time.perf_counter() - start) * 1000
        response = "".join(chunks)
        if not prompt_tokens:
            prompt_tokens = _estimate_tokens(prompt)
        if not completion_tokens:
            completion_tokens = _estimate_tokens(response)

        gen_ms = total_ms - (ttft_ms or 0)
        tps = completion_tokens / (gen_ms / 1000) if gen_ms > 0 else 0.0

        return GenerationResult(
            model=model,
            prompt=prompt,
            response=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            time_to_first_token_ms=ttft_ms,
            total_latency_ms=total_ms,
            tokens_per_second=tps,
        )

    def generate_stream(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> Iterator[str | GenerationResult]:
        """Yield text chunks, then a final GenerationResult."""
        max_tokens = max_tokens or settings.benchmark_max_tokens

        if self.mock:
            result = self._mock_generate(model, prompt, max_tokens)
            words = result.response.split()
            step = max(1, len(words) // 12)
            for i in range(0, len(words), step):
                yield " ".join(words[i : i + step]) + (" " if i + step < len(words) else "")
            yield result
            return

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        start = time.perf_counter()
        ttft_ms: float | None = None
        chunks: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("response"):
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start) * 1000
                        piece = data["response"]
                        chunks.append(piece)
                        yield piece
                    if data.get("prompt_eval_count"):
                        prompt_tokens = data["prompt_eval_count"]
                    if data.get("eval_count"):
                        completion_tokens = data["eval_count"]
                    if data.get("done"):
                        break

        total_ms = (time.perf_counter() - start) * 1000
        response = "".join(chunks)
        if not prompt_tokens:
            prompt_tokens = _estimate_tokens(prompt)
        if not completion_tokens:
            completion_tokens = _estimate_tokens(response)
        gen_ms = total_ms - (ttft_ms or 0)
        tps = completion_tokens / (gen_ms / 1000) if gen_ms > 0 else 0.0

        yield GenerationResult(
            model=model,
            prompt=prompt,
            response=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            time_to_first_token_ms=ttft_ms,
            total_latency_ms=total_ms,
            tokens_per_second=tps,
        )


def hardware_note() -> str:
    system = platform.system()
    machine = platform.machine()
    proc = platform.processor() or "unknown CPU"
    return f"{system} {machine} — {proc}"
