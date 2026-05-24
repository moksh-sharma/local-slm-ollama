"""CLI entrypoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from local_slm.benchmark.report import write_markdown_report
from local_slm.benchmark.runner import benchmark_model, compare_models
from local_slm.chat import run_chat
from local_slm.config import settings
from local_slm.models import ComparisonReport
from local_slm.ollama_client import OllamaClient, hardware_note

console = Console()


def chat_main() -> None:
    parser = argparse.ArgumentParser(description="Offline chat with a local Ollama model")
    parser.add_argument("--model", default=settings.ollama_model)
    args = parser.parse_args()
    run_chat(model=args.model)


def benchmark_main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one model's latency and throughput")
    parser.add_argument("--model", default=settings.ollama_model)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=settings.data_dir / "benchmark_prompts.jsonl",
    )
    args = parser.parse_args()

    client = OllamaClient()
    console.print(f"Benchmarking [cyan]{args.model}[/cyan] (mock={client.mock})…")
    result = benchmark_model(client, args.model, args.prompts)
    print(json.dumps(result.model_dump(), indent=2))


def compare_main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 3 local models on speed + quality (same hardware)"
    )
    parser.add_argument(
        "--models",
        default=settings.benchmark_models,
        help="Comma-separated Ollama model tags",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.results_dir,
    )
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    client = OllamaClient()
    if not client.mock:
        available = set(client.list_models())
        missing = [m for m in models if m not in available and not any(m in a for a in available)]
        if missing:
            console.print("[yellow]Warning:[/yellow] models may not be pulled yet:")
            for m in missing:
                console.print(f"  ollama pull {m}")

    console.print(
        f"Comparing {len(models)} models on [bold]{hardware_note()}[/bold] "
        f"(mock={client.mock})…"
    )
    report = compare_models(models, client=client, output_dir=args.output_dir)

    table = Table(title="Comparison summary")
    table.add_column("Model")
    table.add_column("TTFT")
    table.add_column("Latency")
    table.add_column("tok/s")
    table.add_column("Quality")

    quality_by_model = {q.model: q for q in report.quality}
    for b in report.benchmarks:
        q = quality_by_model.get(b.model)
        table.add_row(
            b.model,
            f"{b.avg_ttft_ms:.0f} ms" if b.avg_ttft_ms else "n/a",
            f"{b.avg_latency_ms:.0f} ms",
            f"{b.avg_tokens_per_second:.1f}",
            f"{q.score:.0%}" if q else "n/a",
        )
    console.print(table)
    console.print(f"\nReports written to {args.output_dir}/latest.md")


def report_main() -> None:
    parser = argparse.ArgumentParser(description="Render markdown from a saved comparison JSON")
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.results_dir / "latest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.results_dir / "latest.md",
    )
    args = parser.parse_args()

    if not args.input.exists():
        console.print(f"[red]Missing {args.input}. Run slm-compare first.[/red]")
        sys.exit(1)

    report = ComparisonReport.model_validate_json(args.input.read_text(encoding="utf-8"))
    write_markdown_report(report, args.output)
    console.print(f"Wrote {args.output}")


if __name__ == "__main__":
    compare_main()
