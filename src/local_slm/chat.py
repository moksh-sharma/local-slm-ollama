"""Interactive offline chat REPL."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from local_slm.config import settings
from local_slm.ollama_client import OllamaClient

console = Console()


def run_chat(model: str | None = None, client: OllamaClient | None = None) -> None:
    model = model or settings.ollama_model
    client = client or OllamaClient()

    mode = "mock" if client.mock else "ollama"
    console.print(
        Panel.fit(
            f"[bold]Local SLM Chat[/bold]\n"
            f"Model: [cyan]{model}[/cyan]  Mode: [green]{mode}[/green]\n"
            "Type [bold]/quit[/bold] to exit, [bold]/model NAME[/bold] to switch.",
            border_style="blue",
        )
    )

    while True:
        try:
            user_input = console.input("[bold green]You>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/quit", "/exit", "quit", "exit"}:
            console.print("Bye.")
            break
        if user_input.startswith("/model "):
            model = user_input.split(maxsplit=1)[1].strip()
            console.print(f"Switched to [cyan]{model}[/cyan]")
            continue

        with console.status("[bold cyan]Generating locally…[/bold cyan]"):
            result = client.generate(model, user_input, max_tokens=512)

        meta = (
            f"{result.total_latency_ms:.0f} ms"
            + (
                f" · TTFT {result.time_to_first_token_ms:.0f} ms"
                if result.time_to_first_token_ms is not None
                else ""
            )
            + f" · {result.tokens_per_second:.1f} tok/s"
        )
        console.print(Panel(Markdown(result.response), title=f"Assistant · {meta}", border_style="dim"))
