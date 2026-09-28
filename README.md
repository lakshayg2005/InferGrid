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

### Results (4 simulated workers, 3 different workloads, 1,330 requests per router; mean with min–max range)

| Router | TTFT p50 | TTFT p99 | Cache hit rate | Tokens recomputed | Busiest worker vs avg |
|---|---|---|---|---|---|
| round_robin | 94 ms (94–94) | 298 ms (284–315) | 79.1% | 66,476 | 1.00× |
| least_loaded | 95 ms (94–95) | 361 ms (283–421) | 79.1% | 66,359 | 1.02× |
| consistent_hash, no bound | 65 ms (64–67) | 1071 ms (826–1479) | 92.5% | 23,905 | 1.36× |
| **consistent_hash, ε = 0.25** | **64 ms (64–64)** | 414 ms (347–502) | 88.2% | 37,516 | 1.07× |

Raw per-request data: [bench/results/](bench/results/). Reproduce with `python scripts/benchmark.py --repeats 3`.

**What this shows**

- Cache-aware routing with bounded loads cuts median time-to-first-token by **32%**
  and the prefill work the cluster does by **44%** compared with round-robin.
- Without the load bound, 27 of the 28 slowest requests landed on a single hot
  worker and waited in its queue (they had a median of only 47 uncached tokens).
  The bound cuts p99 from 1071 ms to 414 ms while keeping the median gain.
- **Open problem:** p99 is still ~40% above round-robin. The slowest requests
  remain queueing delays, not recomputation. The bound counts requests, but at
  ε = 0.25 a busy worker may accept more requests than it has concurrency slots
  while other workers have free ones. Making the bound aware of each worker's
  capacity is the next experiment.

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
