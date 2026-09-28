# InferGrid — Design Document

A distributed LLM serving platform: many LLM worker nodes behind a gateway that
makes them behave like one fast, cheap and fault-tolerant AI service.

## 1. Problem

Companies serving LLMs pay for expensive compute, and naive load balancing wastes it:

1. **Repeated prefill.** Requests often share a prefix (system prompt, RAG context,
   earlier chat turns). A worker that just processed that prefix holds it in its
   KV cache. Round-robin sends the next request elsewhere, so the prefix is
   recomputed from scratch.
2. **Head-of-line blocking.** A 2,000-token generation delays short requests queued
   behind it.
3. **Fragile streams.** A worker crash mid-answer kills the user's response.
4. **Noisy neighbours.** One tenant's burst degrades latency for everyone.
5. **Wasted idle capacity.** Offline work (bulk summarisation, classification) has
   nowhere to go, so clusters sit idle at night and overloaded at peak.

## 2. Goals and non-goals

**Goals**
- Cache-aware routing that measurably beats round-robin on time-to-first-token (TTFT).
- No single point of failure; survive worker and storage node crashes.
- Mid-stream failover: a worker dying does not drop the user's response.
- Per-tenant fairness (rate limiting, load shedding).
- Durable batch inference that soaks up idle capacity.
- Runs on an 8 GB laptop (simulated workers) and deploys to Kubernetes.

**Non-goals**
- Training or fine-tuning models. All models are pre-trained (served by Ollama).
- Writing an inference engine. Workers wrap Ollama or a simulator.
- Byzantine fault tolerance. Nodes may crash or be partitioned, not lie.

## 3. Architecture

```
                 Clients (OpenAI-compatible API, chat UI)
                                  │
                ┌─────────────────▼─────────────────┐
                │   Gateway (stateless, N replicas)  │
                │  rate limit · SWIM · router         │
                │  semantic cache · failover · hedging│
                └──────┬───────────────┬─────────────┘
         interactive   │               │ batch jobs / usage events
                       ▼               ▼
      ┌────────────────────────┐   ┌──────────────────────────┐
      │ Workers (LLM + KV cache)│◄──│ Kafka (Redpanda)          │
      │ SWIM · load shedding     │   │ topics partitioned by     │
      └────────────────────────┘   │ tenant; consumer groups   │
                       │           └──────────────────────────┘
                       ▼
      ┌────────────────────────────────────────────┐
      │ State store (built from scratch)            │
      │ consistent-hash ring · N=3 replicas         │
      │ quorum R/W · hinted handoff · read repair   │
      │ holds: conversations, semantic cache, jobs  │
      └────────────────────────────────────────────┘
```

### 3.1 Gateway
Stateless, so any number of replicas can run behind a plain L4 load balancer.
Exposes `POST /v1/chat/completions` (OpenAI-compatible, SSE streaming) so any
OpenAI SDK works against it unchanged.

### 3.2 Router (the core idea)
- Prompts are tokenised and split into fixed-size **blocks** (16 tokens). Each
  block's hash chains its parent's hash, so equal hashes mean equal prefixes
  (the same trick vLLM uses for its prefix cache).
- The router places the prompt's leading prefix on a **consistent hash ring with
  bounded loads** (Mirrokni et al., Google, 2017): requests sharing a prefix land
  on the same worker, but no worker may exceed `(1 + ε) × average load`. When it
  would, the request walks the ring to the next worker. This trades a little
  cache locality for bounded tail latency.
- **Routing key.** A conversation is identified by its messages up to and
  including the first user message. Every later turn starts with exactly those
  messages, so all turns of a conversation share one key and reach the worker that
  caches its history. Hashing the whole prompt would scatter turns (it changes
  every turn); hashing only the system prompt would send an entire app's traffic
  to one worker.
- **Load view.** Each gateway counts the requests it has in flight on each
  worker, reserving a slot *before* connecting so a burst of simultaneous
  requests cannot all pick the same idle-looking worker. With several
  gateways each sees only its own share, not a cluster-wide total — Phase 3
  adds SWIM (section 3.4) for *liveness*, but load itself still is not
  gossiped between gateways; that remains a known gap for a future phase.
- Baselines for comparison: round-robin (ignores load and cache) and least-loaded
  (perfect balance, ignores cache).

### 3.3 Workers
Wrap an inference backend behind one interface:
- `ollama`: a real pre-trained model (e.g. `qwen2.5:0.5b`) on CPU.
- `sim`: a simulator with a real block-level LRU prefix cache and a latency model
  (`prefill_ms × uncached_tokens + decode_ms × output_tokens`, bounded
  concurrency). Like real engines it also caches the tokens it generates, so the
  next turn can reuse the model's own previous reply. Output is deterministic per
  prompt, so every routing policy is benchmarked on identical conversations.
  Used for benchmarks at a scale a laptop cannot run for real.

Every streamed token carries its **index**. That is what makes failover possible:
the gateway knows exactly how much of the answer the user has already received.

### 3.4 Membership and failure detection
Every worker and the gateway run **SWIM** (Das, Gupta and Motivala, 2002;
`membership/swim.py`), a real implementation over UDP, not a simulation of one:

- Every protocol period (default 500 ms), a member pings one random peer and
  waits for an ack. A missed ping is **not** taken as a crash straight away: a
  few other members are asked to ping the same target on the prober's behalf
  (an indirect ping), since the direct link between just those two members
  might be the only thing broken.
- Only if that also gets no reply is the member marked **suspect**, and only
  **dead** if it does not refute the suspicion (by gossiping a higher
  incarnation number for itself, proving it is still running) within the
  suspicion timeout (default 2 s). Worst case, a real crash takes a few
  seconds to become DEAD; a healthy member routinely refutes far sooner.

  **This margin was tuned up from an initial, tighter version the hard way.**
  A first pass used the textbook-scale timeouts from this project's own unit
  tests (150 ms ping timeout, 1 s suspicion) — those tests run many SWIM nodes
  in one lightly-loaded process, where that is plenty of margin. Run for real
  as separate OS processes competing for CPU while also serving inference
  traffic, those timeouts produced false positives: healthy workers missed a
  ping under ordinary scheduling jitter and got marked dead, and
  `scripts/chaos.py` caught it immediately (`alive_workers` dropped to 1 of 4
  by the end of a run, with no matching kill in the schedule). The fix was
  more realistic timeouts for a real (not unit-tested) environment, plus
  more slack on the indirect-ping budget, which has two hops of network delay
  in addition to the helper's own ping.
- Membership changes piggyback on ping/ack messages already being sent
  (infection-style gossip), so the whole group learns about joins and deaths
  without a central registry and without a separate gossip round.
- Each member advertises metadata (its HTTP URL); the gateway reads
  `alive_http_urls()` from its own SWIM view and only routes to workers it
  currently believes are alive, filtering them out of `router.candidates()`
  before ever attempting a connection.

**Why this matters over just retrying on failure:** without it, every request
routed to an already-dead worker pays a full connect/HTTP-refused round trip
before the gateway's existing failover even gets a chance to try someone else.
`scripts/chaos.py`'s three-way comparison (nothing / failover only / failover +
membership) confirms the mechanism works as designed: mid-stream failovers
(a request that starts on a worker and has to be resumed elsewhere mid-answer)
fall by more than half once membership routes around a worker known to be
dead, rather than finding out partway through an answer.

**What it does not fix, and why that's an interesting result, not a bug:**
in the same chaos runs, *tail latency* with membership on is worse than with
failover alone. This 4-worker cluster is sized for 4 workers' worth of load;
correctly excluding a dead one piles its full share onto the 3 survivors for
the length of the outage, and they queue. "Failover only" coincidentally
avoids this, since its bounded-load router still divides by the stale count
of 4, spreading load thinner even though part of it is wasted on doomed
connection attempts to the dead worker. That is the same root cause as
section 3.2's open problem (the load bound isn't capacity-aware) wearing a
different hat: see section 5 for the numbers and the shared fix.

**A limitation, honestly:** membership only tracks the workers this gateway
was configured with; it does not (yet) drop or add HTTP routing candidates for
members it discovers that were not in the static `--workers` list, so it is a
liveness filter today, not a discovery mechanism.

### 3.5 State store (replication + sharding)
A Dynamo-style leaderless key-value store, written from scratch:

| Mechanism | Purpose |
|---|---|
| Consistent hashing with virtual nodes | Spread keys evenly; move only ~1/N of keys when a node joins |
| Preference list of N=3 replicas | Survive two node failures |
| Tunable quorums (R + W > N) | Choose consistency vs latency per operation |
| Hybrid logical clock versions (vector clocks as stretch goal) | Order concurrent writes |
| Hinted handoff | Accept writes while a replica is down, deliver later |
| Read repair | Heal stale replicas during reads |
| Merkle-tree anti-entropy (stretch) | Background repair of cold keys |

**Why leaderless and not Raft?** Chat history and cache entries favour
availability: a user should still be able to chat during a partition. Rare
conflicting writes to one conversation are acceptable and resolvable. Raft would
make every write wait for one leader and stop writes in a minority partition.

### 3.6 Messaging (Kafka via Redpanda)
- **Batch inference API** (`POST /v1/batches`): jobs go to a topic partitioned by
  tenant. Workers consume only while their interactive queue is short, so batch
  work fills idle capacity without hurting live users. Delivery is at-least-once;
  processing is idempotent (results keyed by job ID), failures retry with
  exponential backoff and jitter, and poison messages go to a dead-letter topic.
- **Usage metering**: every completed request emits a usage event. A billing
  consumer applies them idempotently (keyed by request ID), so redelivery never
  double-charges a tenant.

### 3.7 Load shedding and rate limiting (`gateway/rate_limit.py`, `worker/app.py`)

- **Load shedding (worker-side admission control).** Each backend tracks
  `queue_depth()` (active + waiting requests). `POST /generate` checks this
  *before* starting any work and returns 503 immediately once it reaches
  `max_queue`, instead of adding the request to an ever-growing queue behind
  work that is already running. This needs no new gateway logic: the gateway
  already treats any non-200 response as "try the next candidate"
  (`open_worker_stream`), so a shed request fails over to a less busy worker
  for free, and a cluster where every worker sheds surfaces as one fast 503
  to the client rather than a slow queue-up.
- **Rate limiting (gateway-side, per tenant).** A token bucket per tenant
  (`X-Tenant-Id` header, default `"default"`) refuses a request with 429 and a
  `Retry-After` header once its burst is spent, protecting every other tenant
  sharing the cluster from one tenant's spike. Buckets are in-memory per
  gateway process; with several gateway replicas each enforces its own share
  of the limit rather than one exact cluster-wide number — an exact limit
  needs a store every replica shares (e.g. Redis), the same
  build-it-yourself-vs-Redis trade-off as section 3.5's state store.

### 3.8 Hedged requests (`gateway/app.py::open_hedged_events`)

Targets a different failure mode than mid-stream failover: a worker that is
merely **slow**, not dead (GC-style pause, a burst of decode-heavy neighbours,
a stuck request) — the kind of tail latency admission control alone cannot
fix, since the worker never actually refuses the request.

- The request starts on the router's top candidate as usual. If no token has
  arrived within `hedge_delay_ms`, the gateway *also* starts the request on
  the next candidate and continues with whichever produces a token first,
  discarding the other (closing its connection and releasing its load slot).
- Only the first attempt is hedged; a failure after that is handled by the
  existing mid-stream failover, not a fresh hedge, to keep worst-case worker
  load bounded to 2× a request rather than growing with every retry.
- **A real testing wrinkle worth recording:** httpx's in-process `ASGITransport`
  (used for most of this project's tests, since it needs no real sockets) runs
  a mounted app to completion and buffers the whole response before handing
  anything back — so it cannot show one worker's headers arriving before
  another's body finishes, and cannot exercise hedge timing at all.
  `tests/test_hedging.py` binds real loopback TCP ports with `uvicorn.Server`
  instead, so the race is genuine.

### 3.9 AI features (pre-trained only, no training)
- **Semantic cache**: prompts are embedded with a small pre-trained model
  (`all-minilm` via Ollama). A new prompt within a cosine-similarity threshold of
  a cached one is answered from the cache, which lives in the state store,
  sharded by tenant.
- **Predictive autoscaling**: Holt-Winters forecasting on request rate scales
  workers ahead of spikes (statistical, no training).

## 4. Request flows

**Interactive chat:** client → gateway → rate limit → semantic cache lookup →
router picks worker → stream tokens to client → on worker failure, resume on
another worker from the last token index → store conversation turn → emit usage
event.

**Batch job:** client → gateway → job record written to state store → message
published to Kafka → idle worker consumes → result written to state store →
offset committed → client polls `GET /v1/batches/{id}`.

## 5. Failure handling

| Failure | Behaviour |
|---|---|
| Worker crashes before first token | Gateway retries on next worker on the ring |
| Worker crashes mid-stream | Gateway resumes on another worker from the last received token |
| Worker dead (SWIM-detected) | Excluded from routing candidates before a connection is even attempted |
| Worker slow but not dead | Hedged request to the next candidate after `hedge_delay_ms` with no token |
| Worker at capacity | Sheds the request (503) before queueing it; gateway retries the next candidate |
| Tenant over its rate limit | 429 with `Retry-After`, before any worker is contacted |
| Storage node down | Sloppy quorum + hinted handoff; read repair on recovery |
| Kafka consumer crashes | Uncommitted offsets redelivered; idempotent processing prevents duplicates |

**Mid-stream failover in detail.** The gateway remembers every token it has
forwarded. When a worker's stream breaks (connection reset, error event, or the
stream ending early), it sends the same request to the next worker on the ring
with `resume_text` (the answer so far) and `resume_tokens` (its length). The new
worker prefills the conversation plus the partial answer and numbers its tokens
from `resume_tokens`. The gateway forwards a token only if its index is exactly
the next one expected: lower indices are duplicates and are dropped; a gap is
treated as another failure. At most `max_failovers` moves are made per answer.
`scripts/chaos.py` verifies every surviving answer word for word against the
answer produced without failures.

## 6. Build phases

| Phase | Deliverable |
|---|---|
| 1 | Gateway + workers (sim + Ollama), OpenAI-compatible streaming, round-robin routing |
| 2 | Prefix block hashing, bounded-load consistent hashing, load generator, benchmark vs round-robin |
| 3 | SWIM membership, mid-stream failover, hedged requests, load shedding, rate limiting |
| 4 | Sharded, replicated state store (quorums, hinted handoff, read repair) |
| 5 | Kafka: batch inference API, usage metering, retries, dead-letter topic |
| 6 | Semantic cache, predictive autoscaling |
| 7 | Docker, Kubernetes, Prometheus/Grafana, OpenTelemetry, chaos tests, demo UI |

## 7. How success is measured

Every claim is backed by a benchmark against a baseline:
- TTFT p50/p99 and throughput: cache-aware routing vs round-robin.
- Prefix cache hit rate across the cluster.
- Dropped responses during chaos tests (target: zero).
- Store availability and staleness with nodes killed, for different R/W settings.
- Router overhead per request (it must be negligible next to LLM latency).
