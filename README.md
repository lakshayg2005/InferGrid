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
| 3 | Mid-stream failover, SWIM failure detection, load shedding, rate limiting, hedged requests | ✅ Done |
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
| Gateway | `GET /health` | Router, worker list and which workers SWIM currently reports alive |
| Gateway | `GET /stats` | Requests in flight, routed, failovers, hedges, rate-limit rejections |
| Gateway | `GET /membership` | This gateway's SWIM view (empty if `--swim-port` not set) |
| Worker | `POST /generate` | Internal: stream indexed tokens as SSE; 503s if at capacity (load shedding) |
| Worker | `GET /stats` | Queue depth, prefix-cache hit rate |
| Worker | `GET /membership` | This worker's SWIM view (empty if `--swim-port` not set) |

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

## Chaos test

```bash
python scripts/chaos.py                # ~3.5 minutes
```

Kills a random worker every 12 seconds under load and restarts it 8 seconds later
with an empty cache (a slower recovery than the earlier 3 s default, closer to a
real process supervisor reloading a model), on the same workload and kill
schedule three times: with neither mid-stream failover nor SWIM failure
detection, with failover alone, and with both. Every answer that completes is
checked word for word against the answer the simulator gives when nothing fails.

### Results (4 workers, 5 kills during a 60-second run, identical workload and kill schedule)

| Run | Requests | Answers lost | Answers corrupted | Mid-stream failovers | TTFT p50 | TTFT p99 |
|---|---|---|---|---|---|---|
| nothing (baseline) | 465 | 14 (3.0%) | 0 | 0 | 80 ms | 2746 ms |
| failover only | 504 | 0 (0.0%) | 0 | 17 | 93 ms | 2557 ms |
| failover + membership | 504 | **0 (0.0%)** | 0 | **8** | 159 ms | 3604 ms |

**What this shows**

- Mid-stream failover alone already gets answers to 0% lost, by resuming on
  another worker whenever a crash is noticed. SWIM cuts *how often that noticing
  has to happen at all*: mid-stream failovers fall from 17 to 8, since routing
  now avoids a worker once it's known dead, instead of finding out mid-answer.
- **A real bug found and fixed along the way:** SWIM's first timeouts (150 ms
  ping, 1 s suspicion) were copied from this project's own fast unit tests,
  which run many nodes in one lightly-loaded process. Run for real as separate
  OS processes also serving inference traffic, those timeouts caused false
  positives — a chaos run showed only 1 of 4 workers still marked alive by the
  end, with no matching kill in the schedule. `membership/swim.py`'s defaults
  are now 500 ms / 2 s, with a wider indirect-ping budget; see section 3.4 of
  [DESIGN.md](DESIGN.md) for the full account.
- **Open problem, and it connects to the router's:** tail latency with
  membership is *worse* (3604 ms vs 2557 ms), not better. This cluster has no
  spare capacity — it's sized for 4 workers' worth of load, not 3 — so
  correctly routing around a dead worker piles its full share onto the
  survivors for the length of the outage. "Failover only" coincidentally
  avoids this: its bounded-load math still divides by the stale count of 4,
  spreading load thinner even though some of it is wasted on doomed connection
  attempts to the dead worker. That is the same root cause as Phase 2's open
  problem — the load bound isn't capacity-aware — so fixing it (weighting the
  bound by each worker's `max_concurrency` rather than a flat per-worker cap)
  is the next experiment for both.
- Nothing here is a reliability regression: 0 answers lost or corrupted with
  membership on, in every run. It's a latency effect of running at capacity
  with one fewer worker, not a bug.

## Tests

```bash
pytest
```

## Layout

```
infergrid/
  common/          schemas, tokenizer + prefix block hashing, consistent hash ring, SSE
  membership/      SWIM failure detection over real UDP sockets
  worker/          worker API, prefix cache, load shedding, backends (sim, ollama)
  gateway/         gateway API, routing, mid-stream failover, hedging, rate limiting
  local_cluster.py start/stop worker and gateway processes
  loadgen.py       realistic multi-turn chat workload (open-loop)
scripts/           run_local.py (start a cluster), chat.py (terminal client), benchmark.py, chaos.py
bench/results/     saved benchmark and chaos runs
tests/             tests/test_hedging.py uses real sockets; everything else is fast in-process ASGI
```
