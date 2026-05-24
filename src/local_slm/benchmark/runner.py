"""Run latency/throughput benchmarks and multi-model comparisons."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_slm.benchmark.metrics import summarize_model, summarize_prompt
from local_slm.benchmark.quality import evaluate_task
from local_slm.benchmark.report import save_report, write_markdown_report
from local_slm.config import settings
from local_slm.models import ComparisonReport, ModelQualityReport
from local_slm.ollama_client import OllamaClient, hardware_note


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_model(
    client: OllamaClient,
    model: str,
    prompts_path: Path | None = None,
) -> ModelBenchmark:
    prompts_path = prompts_path or settings.data_dir / "benchmark_prompts.jsonl"
    prompt_rows = _load_jsonl(prompts_path)
    prompt_benches = []

    for row in prompt_rows:
        prompt_id = row["id"]
        category = row.get("category", "generation")
        prompt = row["prompt"]
        max_tokens = row.get("max_tokens", settings.benchmark_max_tokens)
        runs = []

        for _ in range(settings.benchmark_warmup_runs):
            client.generate(model, prompt, max_tokens=max_tokens)

        for _ in range(settings.benchmark_measured_runs):
            runs.append(client.generate(model, prompt, max_tokens=max_tokens))

        prompt_benches.append(summarize_prompt(runs, prompt_id, category))

    return summarize_model(
        model,
        prompt_benches,
        hardware_note=hardware_note(),
        mock=client.mock,
    )


def evaluate_model_quality(
    client: OllamaClient,
    model: str,
    tasks_path: Path | None = None,
) -> ModelQualityReport:
    tasks_path = tasks_path or settings.data_dir / "quality_tasks.json"
    tasks = _load_json(tasks_path)
    results = []

    for task in tasks:
        gen = client.generate(model, task["prompt"], max_tokens=128, temperature=0.0)
        results.append(evaluate_task(task, gen.response))

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    return ModelQualityReport(
        model=model,
        tasks=results,
        passed=passed,
        total=total,
        score=passed / total if total else 0.0,
    )


def compare_models(
    models: list[str] | None = None,
    *,
    client: OllamaClient | None = None,
    output_dir: Path | None = None,
) -> ComparisonReport:
    client = client or OllamaClient()
    models = models or settings.model_list
    output_dir = output_dir or settings.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = [benchmark_model(client, model) for model in models]
    quality = [evaluate_model_quality(client, model) for model in models]

    report = ComparisonReport(
        hardware_note=hardware_note(),
        mock=client.mock,
        benchmarks=benchmarks,
        quality=quality,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"comparison_{stamp}.json"
    md_path = output_dir / f"comparison_{stamp}.md"
    save_report(report, json_path)
    write_markdown_report(report, md_path)
    save_report(report, output_dir / "latest.json")
    write_markdown_report(report, output_dir / "latest.md")

    return report
