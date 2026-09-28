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

### Results (4 simulated workers, 3 different workloads, ~1,330 requests per router; mean with min–max range)

| Router | TTFT p50 | TTFT p99 | Cache hit rate | Tokens recomputed | Busiest worker vs avg |
|---|---|---|---|---|---|
| round_robin | 94 ms (93–95) | 311 ms (266–360) | 79.3% | 66,081 | 1.00× |
| least_loaded | 94 ms (94–95) | 297 ms (280–313) | 79.1% | 66,353 | 1.01× |
| consistent_hash, no bound | 64 ms (64–65) | 1088 ms (799–1544) | 92.5% | 23,905 | 1.36× |
| **consistent_hash, ε = 0.25** | **66 ms (64–69)** | 505 ms (374–673) | 88.1% | 37,580 | 1.08× |

Raw per-request data: [bench/results/](bench/results/). Reproduce with `python scripts/benchmark.py --repeats 3`.
SWIM membership (Phase 3) is deliberately disabled for this benchmark — its own
background traffic is a real cost, but not one this benchmark is measuring; see
`scripts/benchmark.py`'s docstring.

**What this shows**

- Cache-aware routing with bounded loads cuts median time-to-first-token by
  **~30%** and the prefill work the cluster does by **~43%** compared with
  round-robin.
- Without the load bound, the slowest 2% of requests concentrate heavily on a
  single hot worker with a median of only 52 uncached tokens — confirming
  they're queueing delays, not recomputation. The bound cuts p99 by roughly
  half compared with no bound.
- **Open problem, still unresolved:** p99 remains above round-robin's. A
  capacity-aware version of this bound now exists (`Router.set_capacities()`,
  used when SWIM is enabled) and is verified to fix a related failure-mode in
  `scripts/chaos.py` — but this benchmark runs with membership off, by design,
  and even where it's on, the fix targets the "candidate set shrinks" failure
  mode, not this steady-state one. The queueing pattern above is consistent
  with `epsilon` being too generous relative to real worker concurrency
  (`max_concurrency=4` per simulated worker); tightening it, or basing the
  bound on `max_concurrency` even in the no-membership case, is the next
  experiment.

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
| nothing (baseline) | 471 | 16 (3.4%) | 0 | 0 | 80 ms | 2600 ms |
| failover only | 504 | 0 (0.0%) | 0 | 16 | 89 ms | 2767 ms |
| **failover + membership** | 504 | **0 (0.0%)** | 0 | 18 | 94 ms | **2540 ms** |

**What this shows**

- Mid-stream failover alone already gets answers to 0% lost, by resuming on
  another worker whenever a crash is noticed. With membership on top, p99 drops
  a further ~8% (2767 ms → 2540 ms), and 0 answers are lost or corrupted in
  either case, all 5 kills across the run.
- **Two real bugs found and fixed getting to this result, both via this chaos
  test catching a symptom no unit test would have:**
  1. *False positives.* SWIM's first timeouts (150 ms ping, 1 s suspicion) were
     copied from this project's own fast unit tests, which run many nodes in
     one lightly-loaded process. Run for real as separate OS processes also
     serving inference traffic, those timeouts caused healthy workers to be
     marked dead under ordinary scheduling jitter — a chaos run showed only
     1 of 4 workers still marked alive by the end, with no matching kill in
     the schedule. Fixed by loosening the defaults to 500 ms / 2 s with a
     wider indirect-ping budget.
  2. *Stuck-dead restarts.* A restarted worker is a brand-new `SwimNode` with
     no memory of its old incarnation number. Per SWIM's own merge rule, an
     ALIVE claim at the same incarnation a peer last recorded as DEAD is
     rejected — so a revived worker could stay marked dead forever unless it
     got lucky and heard a peer re-gossip its death within a narrow window (a
     race, not a guarantee). A chaos run showed `alive_workers` stuck at 2 of 4
     long after every kill should have recovered. Fixed by seeding a node's
     incarnation from the current time rather than 0, which is essentially
     guaranteed to exceed anything a peer last remembered.
  3. *A capacity-aware load bound was also added* (`Router.set_capacities()`,
     `gateway/router.py`): when membership is on, each worker gossips its real
     `max_concurrency`, and the router's bound is anchored to a worker's own
     capacity rather than the cluster average — so losing a worker no longer
     silently raises what the survivors are allowed to carry. This is what
     turned membership's tail-latency effect from worse to better.
- `tests/test_swim.py::test_a_restarted_node_...` is a genuine regression test
  for bug 2: it force-exhausts the "lucky" refutation path and is confirmed to
  fail on the pre-fix code. See DESIGN.md section 3.4 for the full account.
- This is a single run, not averaged over repeated seeds like the routing
  benchmark above; the exact numbers have shown real run-to-run variance
  across earlier attempts documented in git history, though the *shape* of the
  result (0 lost/corrupted, membership improving rather than hurting p99) has
  been consistent since both fixes landed.

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
