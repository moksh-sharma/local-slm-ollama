"""Aggregate benchmark statistics."""

from __future__ import annotations

from local_slm.models import GenerationResult, ModelBenchmark, PromptBenchmark


def _avg(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def summarize_prompt(runs: list[GenerationResult], prompt_id: str, category: str) -> PromptBenchmark:
    bench = PromptBenchmark(
        prompt_id=prompt_id,
        category=category,
        runs=runs,
        avg_ttft_ms=_avg([r.time_to_first_token_ms for r in runs]),
        avg_latency_ms=sum(r.total_latency_ms for r in runs) / len(runs),
        avg_tokens_per_second=sum(r.tokens_per_second for r in runs) / len(runs),
        avg_completion_tokens=sum(r.completion_tokens for r in runs) / len(runs),
    )
    return bench


def summarize_model(
    model: str,
    prompts: list[PromptBenchmark],
    *,
    hardware_note: str = "",
    mock: bool = False,
) -> ModelBenchmark:
    return ModelBenchmark(
        model=model,
        hardware_note=hardware_note,
        mock=mock,
        prompts=prompts,
        avg_ttft_ms=_avg([p.avg_ttft_ms for p in prompts]),
        avg_latency_ms=sum(p.avg_latency_ms for p in prompts) / max(len(prompts), 1),
        avg_tokens_per_second=sum(p.avg_tokens_per_second for p in prompts) / max(len(prompts), 1),
    )
