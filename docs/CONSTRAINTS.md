# Privacy, latency, and cost — why local SLMs matter

This project treats **local inference as an engineering tradeoff**, not a free upgrade over cloud APIs.

## Privacy

| Cloud API | Local (Ollama) |
|-----------|----------------|
| Prompts traverse the network | Prompts stay on your machine |
| Vendor may log, retain, or train on data (see their DPA/ToS) | You control disk, backups, and network egress |
| Compliance = trust + contract | Compliance = your infra boundaries |

**When local wins:** regulated data (HR, health, legal), customer IP, air-gapped environments, or any workflow where "we don't send it out" is a hard requirement.

**When cloud is fine:** public data, low sensitivity, or you already have an enterprise agreement with strict zero-retention settings.

## Latency

Two numbers matter for UX:

1. **Time to first token (TTFT)** — how long until the user sees *anything*. Drives perceived responsiveness in chat.
2. **Total latency** — TTFT + generation time for the full answer.

Local models avoid network RTT to a remote region, but **model load and prefill on consumer hardware** can still make TTFT worse than a fast hosted API on a cold start.

This repo measures both via streaming (`/api/generate` with `stream: true`). Warmup runs (`BENCHMARK_WARMUP_RUNS`) isolate steady-state performance from cold-start effects.

## Cost

| Cost type | Cloud API | Local |
|-----------|-----------|-------|
| Marginal per 1M tokens | $0.15–$15+ depending on model | ~$0 API fee |
| Fixed | None | GPU/RAM, electricity, engineer time |
| Scaling | Pay per use | Buy more hardware |

**Break-even intuition:** heavy daily usage on a mid-size model often beats API spend within months on owned hardware; sporadic usage rarely does.

## Quality vs speed

We compare three **small** models on the **same machine**:

| Model | Role in comparison |
|-------|---------------------|
| `llama3.2:1b` | Speed tier — lowest latency, weakest reasoning |
| `llama3.2:3b` | Balanced — default chat model |
| `qwen2.5:3b` | Quality tier at ~3B params — stronger structured output |

Parameter count is not the whole story (architecture, quantization, and tokenizer matter), but on identical hardware smaller quants usually win speed and lose accuracy on math/instruction tasks.

Run `slm-compare` to produce `data/results/latest.md` with measured TTFT, throughput, and task pass rates on **your** silicon — numbers from a README table are not portable.

## Decision checklist

Use **local** when privacy or offline operation is non-negotiable, volume is high enough to amortize hardware, or latency to *your* GPU beats round-trips to a distant API region.

Use **cloud** when you need the largest models, burst capacity without capital expense, or managed safety/compliance features.

Use **hybrid** (common in production): local SLM for routing, PII redaction, or draft answers; cloud for hard reasoning steps — with explicit data-handling boundaries.
