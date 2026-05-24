# Local SLM App with Ollama

Run small language models **entirely offline**, benchmark inference on your hardware, and compare **three models** on speed and quality. Built to make privacy, latency, and cost tradeoffs explicit — not hand-wavy.

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────────┐
│  slm-chat   │────▶│    Ollama    │────▶│  Local model weights    │
│  (REPL)     │     │  localhost   │     │  (no cloud API calls)   │
└─────────────┘     └──────────────┘     └─────────────────────────┘
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
            slm-benchmark      slm-compare
            (one model)        (3 models → report)
                   │                 │
                   └────────┬────────┘
                            ▼
                   TTFT · latency · tok/s · quality score
```

## Quick start

```bash
cd local-slm-ollama
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Try without Ollama (synthetic timings + mock answers)
export MOCK_OLLAMA=true
slm-compare

# Interactive chat
slm-chat
```

### With Ollama (real offline inference)

1. Install [Ollama](https://ollama.com) and start the daemon.
2. Pull the three comparison models (≈2–5 GB each depending on quant):

```bash
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
```

3. Run the full comparison on **your** hardware:

```bash
cp .env.example .env
export MOCK_OLLAMA=false
slm-compare
```

Reports land in `data/results/latest.md` and `latest.json`.

## Commands

| Command | Purpose |
|---------|---------|
| `slm-chat` | Offline REPL with per-turn latency stats |
| `slm-benchmark --model llama3.2:3b` | Latency/throughput for one model |
| `slm-compare` | Benchmark + quality eval for 3 models |
| `slm-report` | Re-render markdown from saved JSON |

Configure via `.env`: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `BENCHMARK_MODELS`, warmup/measured run counts.

## What gets measured

### Speed (same prompts, same machine)

- **TTFT** — time to first streamed token (chat UX)
- **Total latency** — end-to-end generation time
- **Throughput** — completion tokens / second
- Warmup runs before measured runs to separate cold-start noise

Prompts live in `data/benchmark_prompts.jsonl` (short / medium / long generation).

### Quality (deterministic, no judge LLM)

Fixed tasks in `data/quality_tasks.json`: arithmetic, JSON shape, summarization keywords, instruction format. Scored pass/fail so comparisons are reproducible.

## Three-model comparison

Default lineup (edit `BENCHMARK_MODELS` to swap e.g. `phi3:mini`):

| Model | Typical role |
|-------|----------------|
| `llama3.2:1b` | Fastest — best TTFT / tok/s, weakest on hard tasks |
| `llama3.2:3b` | Balanced default for chat |
| `qwen2.5:3b` | Stronger quality at similar size — pays in latency |

After `slm-compare`, read the generated tradeoff section in the report. See [docs/CONSTRAINTS.md](docs/CONSTRAINTS.md) for privacy, latency, and cost framing.

## Project layout

| Path | Purpose |
|------|---------|
| `src/local_slm/ollama_client.py` | Streaming Ollama client + mock mode |
| `src/local_slm/benchmark/` | Runner, metrics, quality checks, reports |
| `src/local_slm/chat.py` | Interactive offline chat |
| `data/benchmark_prompts.jsonl` | Standard speed prompts |
| `data/quality_tasks.json` | Quality eval tasks |
| `data/results/` | Comparison JSON + markdown output |
| `docs/CONSTRAINTS.md` | Privacy / latency / cost analysis |

## Tests

```bash
pytest -q
```

## License

MIT
