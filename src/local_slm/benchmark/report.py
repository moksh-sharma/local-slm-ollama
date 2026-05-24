"""Markdown and JSON report writers."""

from __future__ import annotations

import json
from pathlib import Path

from local_slm.models import ComparisonReport, ModelBenchmark, ModelQualityReport


def save_report(report: ComparisonReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f} ms"


def _fmt_tps(value: float) -> str:
    return f"{value:.1f} tok/s"


def _speed_table(benchmarks: list[ModelBenchmark]) -> str:
    lines = [
        "| Model | Avg TTFT | Avg latency | Throughput |",
        "|-------|----------|-------------|------------|",
    ]
    for b in sorted(benchmarks, key=lambda x: x.avg_tokens_per_second, reverse=True):
        lines.append(
            f"| `{b.model}` | {_fmt_ms(b.avg_ttft_ms)} | {_fmt_ms(b.avg_latency_ms)} | "
            f"{_fmt_tps(b.avg_tokens_per_second)} |"
        )
    return "\n".join(lines)


def _quality_table(quality: list[ModelQualityReport]) -> str:
    lines = [
        "| Model | Score | Passed | Notes |",
        "|-------|-------|--------|-------|",
    ]
    for q in sorted(quality, key=lambda x: x.score, reverse=True):
        failed = [t.task_id for t in q.tasks if not t.passed]
        notes = ", ".join(failed) if failed else "all tasks passed"
        lines.append(f"| `{q.model}` | {q.score:.0%} | {q.passed}/{q.total} | {notes} |")
    return "\n".join(lines)


def _tradeoff_section(report: ComparisonReport) -> str:
    if not report.benchmarks or not report.quality:
        return "_No data._"

    fastest = max(report.benchmarks, key=lambda b: b.avg_tokens_per_second)
    best_quality = max(report.quality, key=lambda q: q.score)
    balanced = report.benchmarks[1].model if len(report.benchmarks) > 1 else fastest.model

    return f"""\
### Quality vs speed (same hardware)

| Constraint | What we measured | Takeaway |
|------------|------------------|----------|
| **Latency** | Time-to-first-token (TTFT) + total generation time | Smaller models ({fastest.model}) respond faster — critical for interactive chat. |
| **Throughput** | Completion tokens / second | Higher tok/s lowers cost-per-answer on owned hardware (electricity + amortized GPU). |
| **Quality** | Deterministic task pass rate on fixed prompts | {best_quality.model} scored highest ({best_quality.score:.0%}) but may be slower. |
| **Privacy** | All inference local via Ollama | Prompts never leave the machine — no vendor logging or retention policy. |
| **Cost** | $0 marginal per token vs cloud APIs | You pay upfront in RAM/VRAM and power; break-even vs API depends on volume. |

**Practical pick on this machine:** `{balanced}` is the usual compromise when TTFT and task accuracy both matter.
"""


def write_markdown_report(report: ComparisonReport, path: Path) -> None:
    mode = "MOCK (synthetic timings)" if report.mock else "LIVE (Ollama)"
    body = f"""# Local SLM comparison report

- **Generated:** {report.generated_at}
- **Hardware:** {report.hardware_note}
- **Mode:** {mode}

## Speed benchmark

{_speed_table(report.benchmarks)}

## Quality benchmark

{_quality_table(report.quality)}

## Per-prompt latency

"""
    for bench in report.benchmarks:
        body += f"### `{bench.model}`\n\n"
        body += "| Prompt | Category | Avg TTFT | Avg latency | tok/s |\n"
        body += "|--------|----------|----------|-------------|-------|\n"
        for p in bench.prompts:
            body += (
                f"| {p.prompt_id} | {p.category} | {_fmt_ms(p.avg_ttft_ms)} | "
                f"{_fmt_ms(p.avg_latency_ms)} | {_fmt_tps(p.avg_tokens_per_second)} |\n"
            )
        body += "\n"

    body += _tradeoff_section(report)
    body += "\n\n_Re-run with `slm-compare` after pulling models on your hardware._\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
