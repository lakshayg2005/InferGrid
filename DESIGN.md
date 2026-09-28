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
                │  auth · rate limit · load shedding │
                │  semantic cache · router · failover│
                └──────┬───────────────┬─────────────┘
         interactive   │               │ batch jobs / usage events
                       ▼               ▼
      ┌────────────────────────┐   ┌──────────────────────────┐
      │ Workers (LLM + KV cache)│◄──│ Kafka (Redpanda)          │
      │ gossip membership+load  │   │ topics partitioned by     │
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
- **Load view.** In Phase 2 each gateway counts the requests it has in flight on
  each worker, reserving a slot *before* connecting so a burst of simultaneous
  requests cannot all pick the same idle-looking worker. With several gateways
  each sees only its own share; Phase 3 switches to load reported by workers.
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
SWIM-style gossip (ping, indirect ping-req, suspicion) so workers and gateways
learn about joins and failures without a central registry.

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

### 3.7 AI features (pre-trained only, no training)
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
| Worker slow | Hedged request to a second worker after p95 latency |
| Storage node down | Sloppy quorum + hinted handoff; read repair on recovery |
| Cluster overloaded | Admission control sheds lowest-priority work first (batch before interactive) |
| Kafka consumer crashes | Uncommitted offsets redelivered; idempotent processing prevents duplicates |

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
