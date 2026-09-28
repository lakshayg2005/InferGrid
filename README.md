# InferGrid

A distributed LLM serving platform. A gateway sits in front of many LLM workers and
makes them behave like one fast, cheap and fault-tolerant AI service: cache-aware
routing, a sharded and replicated state store, Kafka-based batch inference, and
streams that survive worker crashes.

See [DESIGN.md](DESIGN.md) for the architecture and the reasoning behind it.

## Status

| Phase | What | Status |
|---|---|---|
| 1 | Gateway + workers (simulated + Ollama), OpenAI-compatible streaming, round-robin routing, failover before first token | ✅ Done |
| 2 | Cache-aware routing (prefix hashing + bounded-load consistent hashing), benchmarks | ✅ Done |
| 3 | Gossip membership, mid-stream failover, hedging, load shedding, rate limiting | Next |
| 4 | Sharded, replicated state store | |
| 5 | Kafka: batch inference, usage metering | |
| 6 | Semantic cache, predictive autoscaling | |
| 7 | Kubernetes, observability, chaos tests, demo UI | |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -e ".[dev]"

python scripts/run_local.py --workers 3     # terminal 1: 3 simulated workers + gateway (cache-aware router)
python scripts/chat.py                      # terminal 2: chat with the cluster
```

The gateway listens on `http://127.0.0.1:8700` and speaks the OpenAI API, so any
OpenAI client works:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8700/v1", api_key="unused")
client.chat.completions.create(model="infergrid", messages=[{"role": "user", "content": "Hi"}])
```

### With a real model (no GPU needed)

```bash
ollama serve                         # if Ollama is not already running
ollama pull qwen2.5:0.5b             # ~400 MB, runs on CPU
python scripts/run_local.py --workers 2 --backend ollama
```

## Endpoints

| Service | Endpoint | Purpose |
|---|---|---|
| Gateway | `POST /v1/chat/completions` | OpenAI-compatible chat, streaming or not |
| Gateway | `GET /health` | Router and worker list |
| Gateway | `GET /stats` | Requests in flight and routed per worker |
| Worker | `POST /generate` | Internal: stream indexed tokens as SSE |
| Worker | `GET /stats` | Queue depth, prefix-cache hit rate |

Every response carries `X-InferGrid-Worker` (which worker served it) and `X-Request-Id`.

## Benchmark

```bash
python scripts/benchmark.py            # ~6 minutes; results saved to bench/results/
```

Every routing policy gets a fresh 4-worker cluster and the identical workload
(multi-turn chats sharing long system prompts, open-loop Poisson arrivals).

It compares round-robin, least-loaded, and consistent hashing with and without the
load bound, reporting time-to-first-token percentiles, prefix-cache hit rate, tokens
recomputed, and load imbalance.

## Tests

```bash
pytest
```

## Layout

```
infergrid/
  common/          schemas, tokenizer + prefix block hashing, consistent hash ring, SSE
  worker/          worker API, prefix cache, backends (sim, ollama)
  gateway/         gateway API, load tracking, routing policies
  local_cluster.py start/stop worker and gateway processes
scripts/           run_local.py (start a cluster), chat.py (terminal client), benchmark.py
bench/results/     saved benchmark runs
tests/
```
