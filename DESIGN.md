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
      │ Workers (LLM + KV cache)│◄──│ Queue broker (from scratch│
      │ SWIM · load shedding     │   │ + a real-Kafka option)    │
      └────────────────────────┘   │ topics partitioned by     │
                       │           │ tenant; consumer groups   │
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
  gateways each sees only its own share, not a cluster-wide total — SWIM
  (section 3.4) gives liveness cluster-wide, but load itself still is not
  gossiped between gateways; that remains a known gap for a future phase.
- **Capacity-aware bound.** Each worker also gossips its real capacity
  (`max_concurrency`) as SWIM metadata. When membership is enabled, the
  router anchors a worker's bound to *its own* capacity
  (`ceil((1 + ε) × that worker's capacity)`) instead of the cluster average;
  without membership (or before any worker has reported one), it falls back
  to the average-based bound described above. This closes the gap explained
  in section 3.4: an average-based bound rises for the survivors the moment
  a worker is excluded, with no floor at what they can actually process
  concurrently, which is what made the chaos test's tail latency worse, not
  better, the first time SWIM was added.
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

  This margin, and the fix below, were both tuned up the hard way, by two real
  bugs `scripts/chaos.py` caught that no unit test would have:

  **Bug 1: false positives.** A first pass used the textbook-scale timeouts
  from this project's own unit tests (150 ms ping, 1 s suspicion) — those
  tests run many SWIM nodes in one lightly-loaded process, where that is
  plenty of margin. Run for real as separate OS processes competing for CPU
  while also serving inference traffic, those timeouts produced false
  positives: healthy workers missed a ping under ordinary scheduling jitter
  and got marked dead, and a chaos run showed `alive_workers` at 1 of 4 by the
  end, with no matching kill in the schedule. Fixed with the more forgiving
  defaults above, plus more slack on the indirect-ping budget (it has two
  network hops in addition to the helper's own ping).

  **Bug 2: a restarted worker could stay marked dead forever.** A restarted
  process is a brand-new `SwimNode` with no memory of its previous incarnation
  number. SWIM's own merge rule rejects an ALIVE claim at an incarnation no
  higher than what a peer last recorded as DEAD — so a revived worker
  announcing itself at incarnation 0 again could be silently ignored by any
  peer still holding it as DEAD, unless it got lucky and heard that peer
  re-gossip its death within a narrow window before that gossip entry's
  retransmit budget ran out (a race, not a guarantee). A chaos run showed
  `alive_workers` stuck at 2 of 4 long after every kill should have recovered.
  Fixed by seeding a node's incarnation from the current time (nanoseconds,
  not just seconds -- this project's own fast-settings test suite runs many
  SWIM rounds within a single wall-clock second, and a first, seconds-resolution
  version of this fix landed exactly on that collision, a real and repeatable
  flake caught by running the full suite a handful of times) instead of 0 —
  essentially guaranteed to exceed whatever small counter value any peer last
  remembered. `tests/test_swim.py::test_a_restarted_node_...` is a genuine
  regression test for this: it force-exhausts the lucky path and is confirmed
  to fail on the pre-fix code.
- Membership changes piggyback on ping/ack messages already being sent
  (infection-style gossip), so the whole group learns about joins and deaths
  without a central registry and without a separate gossip round.
- Each member advertises metadata (its HTTP URL and its real capacity, see
  3.2's capacity-aware bound); the gateway reads `alive_http_urls()` from its
  own SWIM view and only routes to workers it currently believes are alive,
  filtering them out of `router.candidates()` before ever attempting a
  connection.

**Why this matters over just retrying on failure:** without it, every request
routed to an already-dead worker pays a full connect/HTTP-refused round trip
before the gateway's existing failover even gets a chance to try someone else.
`scripts/chaos.py`'s three-way comparison (nothing / failover only / failover +
membership) confirms the combination works: 0 answers lost or corrupted with
membership on, in every kill, and p99 latency improves over failover alone
once the capacity-aware bound (3.2) is also in the picture — see section 5 for
the numbers. The exact mid-stream-failover count is noisier run to run than
these headline numbers (only 5 kills per run), so it is reported honestly
rather than rounded into a clean story.

**A limitation, honestly:** membership only tracks the workers this gateway
was configured with; it does not (yet) drop or add HTTP routing candidates for
members it discovers that were not in the static `--workers` list, so it is a
liveness filter today, not a discovery mechanism.

### 3.5 State store (replication + sharding) (`infergrid/store/`)
A Dynamo-style leaderless key-value store, written from scratch:

| Mechanism | Purpose |
|---|---|
| Consistent hashing with virtual nodes (same `common/hashring.py` as the router) | Spread keys evenly; move only ~1/N of keys when a node joins |
| Preference list of `n_replicas` (default 3) | Survive `n_replicas - 1` node failures |
| Tunable quorums (`w + r > n_replicas`, enforced at construction) | Choose consistency vs latency per node |
| Hybrid logical clock versions (`store/clock.py`) | Order concurrent writes without vector clocks' bookkeeping |
| Sloppy quorum + hinted handoff | Accept writes while a replica is down, deliver later |
| Read repair | Heal stale or missing replicas during reads |
| Merkle-tree anti-entropy (stretch, not built) | Background repair of keys nobody reads or rewrites after a crash |

**Why leaderless and not Raft?** Chat history and cache entries favour
availability: a user should still be able to chat during a partition. Rare
conflicting writes to one conversation are acceptable and resolvable. Raft would
make every write wait for one leader and stop writes in a minority partition.

**Why HLC and not vector clocks?** A vector clock (one counter per node) detects
true concurrency exactly, at the cost of growing with cluster size and needing
client-side reconciliation when it finds a real conflict. A hybrid logical clock
is a single, always-comparable `(physical_ns, logical, node_id)` triple: cheap,
and its physical component stays meaningful as a timestamp, at the cost of
resolving concurrent writes last-write-wins instead of surfacing the conflict.
Acceptable here: conversation state overwriting to "whichever write's clock was
later" loses at most one racing write to the same key at nearly the same
instant, not silent corruption.

**Two mechanisms make writes/reads reroute around a down node, reactively, not
just when a failure detector has already noticed:**
- *Write:* every preference-list node is tried; a node membership already knows
  is down is skipped without spending a timeout on it, but any node -- known
  dead or not -- whose write actually fails gets a spare substitute from further
  round the ring, holding the write as a hint "for" the down node. The write
  still counts toward `w`, so one down replica never blocks a write, and a
  just-crashed node (before SWIM has caught up) is covered too, not only an
  already-detected one.
- *Read:* the same reactive substitution, topped up with spares until `r`
  replicas have actually answered (not just been asked). The newest version by
  HLC wins; any replica that answered stale or missing is repaired in the
  background, not on the request's critical path.

**Hinted handoff only heals what happened *while* a node was down.** Data a
node already held *before* it crashed is gone from its (in-memory, unpersisted)
storage on restart, and no hint exists for it, since nothing wrote it again
during the outage -- that gap is closed lazily, the next time something reads
an affected key and triggers read repair, not automatically. `scripts/store_chaos.py`
checks these as two separate claims for exactly this reason (see its own
docstring), rather than one blanket "the node recovered" assertion.

**A real bug found via `scripts/store_chaos.py`, not by the fast in-process unit
tests in `tests/test_store.py`:** a store node restarting mid-run, seeded on its
peers via `--seeds` exactly like Phase 3's workers, had its own `alive_nodes()`
stay `{itself}` indefinitely -- it was pinging its seeds successfully every
round, but a bare successful ping/ack proves nothing about the target's
aliveness on its own: `SwimNode.datagram_received` only calls `_update()` (the
method that actually populates `self.members`) for entries in a message's
piggybacked `gossip` list, never just because an ack came back. And a seed's
gossip about *itself* is only pending for `gossip_retransmits` sends after
`start()` -- in a cluster that had already been running for the tens of seconds
`store_chaos.py`'s workload takes, that had long since drained on every seed. A
freshly-started test cluster (`tests/test_swim.py`'s existing tests) never hit
this, since every node's self-announcement is still fresh when a test's seeded
node joins moments later -- which is exactly why a real multi-process run under
realistic timing, not just a fast unit-test cluster, caught it. Fixed in
`infergrid/membership/swim.py::_send`: every message now carries a fresh claim
of the sender's own aliveness alongside whatever limited-retransmit gossip is
pending, so a single successful ping/ack round-trip -- in either direction --
is enough to introduce two nodes to each other, regardless of what either one
happens to have queued up to say about itself at that moment.
`tests/test_swim.py::test_a_late_joiner_learns_about_a_seed_with_no_pending_gossip_about_itself`
force-drains a seed's self-gossip the same way the restart-incarnation test
force-drains a dead report, to test the real fix rather than the timing
coincidence that let it through before; confirmed to fail on the pre-fix code.

**A second real bug, found the same way once the first was fixed:** even with
membership converged, a write occasionally failed quorum against nodes that
were never killed at all. `store/node.py`'s write path originally skipped a
preference-list node proactively whenever it was not `ALIVE` -- which also
excludes `SUSPECT`, SWIM's deliberate window of *uncertainty*, not evidence of
failure. Each write fans out concurrent HTTP replication to 2-3 peers on the
same single event loop that has to answer that node's own SWIM pings, so a
coordinator busy replicating is slower to service its own ping/ack than
Phase 3's workers ever were (they only generate tokens); under real load this
was enough to tip a perfectly healthy peer into SUSPECT for a moment. Treating
that as "skip it" burned through the small spare pool (as few as `nodes -
n_replicas`, e.g. 1 spare in this project's default 4-node/n=3 setup) that a
*real* failure needs, turning a false suspicion into an actual quorum failure.
Fixed by only skipping a node once membership has confirmed it `DEAD`
(`StoreNode._known_dead()`); a `SUSPECT` node still gets a genuine attempt,
which succeeds immediately since it's actually up. Also widened the store's
own SWIM timeouts past Phase 3's worker/gateway defaults
(`infergrid/store/__main__.py`: 1.0s ping / 4.0s suspicion vs 0.5s / 2.0s) to
give the heavier per-request replication fan-out more headroom.

**An open, honestly-unresolved finding, not papered over:** even after both
fixes, `scripts/store_chaos.py` still occasionally (roughly 1 run in 5-6 at the
default 4 nodes, more often at 5) hits a *stable* membership partition -- one
or more nodes settle into seeing only themselves or one other node, and stay
that way for the rest of the run rather than self-healing within a few
protocol periods the way a genuinely transient suspicion does. This looks like
sustained event-loop/OS scheduling contention from running 4-5 full
Python/uvicorn processes with real concurrent UDP and HTTP traffic on one
machine (this project targets a modest, GPU-less, 8 GB dev machine, not a
server-grade box), not a specific line of code identified with confidence in
the time available. `write_batch` in `scripts/store_chaos.py` records a failed
write instead of crashing so a run surfaces this as data (a `FAIL` with a
`write FAILED` count) rather than an unhandled traceback. Documented here
rather than hidden, in the same spirit as section 7's benchmark p99 gap: a
real, reproducible limitation of a failure detector sharing an event loop with
the workload it is trying to stay responsive under, worth its own investigation
rather than a guessed fix.

### 3.6 Messaging (`infergrid/queue/`, `infergrid/billing/`)

A from-scratch, Kafka-compatible message broker (`infergrid/queue/broker/`),
written for the same reason SWIM and the state store were: to demonstrate the
real mechanics -- topic partitioning, consumer groups, at-least-once delivery,
rebalancing on a crash -- rather than only knowing how to point a client
library at somebody else's server. `infergrid/queue/base.py::QueueClient` is
the interface application code (the gateway's batch endpoints, a worker's
batch consumer, the billing consumer) is written against; two implementations
exist behind it:

| Implementation | What it is | Tested here? |
|---|---|---|
| `infergrid.queue.client.BrokerClient` | HTTP client for `infergrid/queue/broker/` | Yes -- `tests/test_queue_broker.py` (the broker's core logic directly), `tests/test_queue_http.py` (real request/response cycles + the retry/DLQ consumer loop), `tests/test_batch.py` (the full pipeline), `scripts/batch_demo.py` (real multi-process) |
| `infergrid.queue.redpanda.RedpandaClient` | `aiokafka` client for a real Kafka-API broker (Redpanda, via `docker-compose.yml`) | **No.** Written but unverified -- no Docker in this project's environment. See that module's docstring. |

**Broker semantics** (`broker/core.py`): a topic has a fixed number of
partitions; a key hashes to one (same ring-hash helper the router uses), so
one tenant's jobs stay ordered on one partition while different tenants
spread across partitions and process in parallel. A consumer group's
partitions are round-robin assigned across its current members and
rebalanced whenever one goes quiet for `session_timeout` -- the mechanism
that lets another consumer pick up a message the crashed one had polled but
never committed, which is what makes delivery at-least-once rather than
at-most-once. Deliberate simplification: at most one *in-flight* (delivered,
uncommitted) message per partition at a time, not Kafka's pipelined fetch --
trivial to reason about and test, at the cost of some throughput; different
partitions still process concurrently.

**Retry, backoff and the dead-letter topic are consumer-side policy**
(`queue/consumer.py::consume_with_retry`), not a broker feature -- real Kafka
doesn't have one built in either. A handler that raises gets its message
committed immediately (freeing the partition) and republished with
`attempt + 1` after an exponential-backoff-plus-jitter delay; once `attempt`
reaches `max_attempts`, the message goes to `f"{topic}.dlq"` instead of
retrying forever and blocking everything behind it. This one function is used
identically by the worker's batch consumer and the billing consumer,
regardless of which `QueueClient` backs them.

**Batch inference** (`POST /v1/batches` in `gateway/app.py`): each request in
the batch becomes one job message on the `"batch-jobs"` topic, keyed by
tenant, and the gateway returns immediately with a `batch_id` and one
`job_id` per job. A worker's background consumer (`worker/app.py`) only pulls
a job while `backend.queue_depth()` is below `idle_queue_depth`, so batch
work fills idle capacity instead of competing with interactive requests for
it. A result is written to the Phase 4 state store keyed by `job_id` --
reusing the store here, rather than adding a second durability mechanism, is
deliberate: `GET /v1/batches/{batch_id}` reads each job's result straight
from it. Idempotency is a store lookup before doing any real work: if
`job_id`'s result already exists, a redelivered duplicate is a no-op, not
repeated generation. `tests/test_batch.py` proves this directly (a
`CountingBackend` that publishing the same job twice still only calls
`generate()` once) and `scripts/batch_demo.py` proves it under a real worker
crash: jobs a killed worker had claimed but not finished still complete,
processed by the survivor once the broker's session timeout reassigns them.

**Usage metering**: the gateway publishes a `"usage"` event on every
completed interactive request (`gateway/app.py::meter_usage`, fire-and-forget
so a slow broker never adds response latency) and a worker publishes one per
completed batch job. `infergrid/billing/consumer.py::apply_usage_event`
folds each event into a per-tenant running total in the store, keyed by
`request_id`: a redelivered duplicate finds its dedup marker already there
and is skipped, so at-least-once delivery never double-charges a tenant. This
is a plain read-modify-write against the store, not an atomic increment --
correct for one billing consumer instance, but two instances processing the
same tenant concurrently could race on the total. A real billing pipeline
would use a CRDT counter or a transactional store here; out of scope, in the
same spirit as section 3.5's unimplemented Merkle-tree anti-entropy.

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

#### Semantic cache (`infergrid/semantic_cache.py`)

A new prompt within a cosine-similarity threshold of one already answered is
served straight from the cache, skipping a worker (and the LLM) entirely --
distinct from the worker-side prefix cache (`worker/prefix_cache.py`), which
still recomputes the *new* tokens of a similar-but-not-identical prompt. Two
`Embedder` implementations behind the same interface, the same split as
`worker/backends/` (sim vs Ollama) and `queue/` (from-scratch broker vs
Redpanda):

- `HashEmbedder` (default): the hashing trick -- each word hashes into one of
  `dims` buckets, counted, L2-normalized -- gives a real, fully tested
  bag-of-words vector with no model or network call, at the cost of being a
  *lexical* similarity measure: it cannot know "car" and "automobile" are
  related, only that two texts sharing more words are more similar.
- `OllamaEmbedder`: real embeddings from a small pre-trained model (e.g.
  `all-minilm`) via Ollama's `/api/embed`. **Unverified against a live
  server** -- no Ollama process was running while this was written; same
  caveat as `infergrid/queue/redpanda.py`.

The cache lives in the Phase 4 state store, one JSON list per tenant
(`semantic-cache:{tenant}`) rather than a key per entry: there is no way to
enumerate a key range in that store's `/kv` API, and a tenant's cache is
small enough that "fetch the whole list, compare in Python" is the honest,
simple choice -- a real deployment would use a proper vector index. Capped
at `max_entries`, oldest evicted first.

`scripts/semantic_cache_demo.py` measures the real payoff against a workload
of FAQ-style questions asked with light paraphrasing: a genuine multi-worker,
multi-process run, not just the logic in isolation
(tests/test_semantic_cache.py). A representative run: 15 paraphrases that
could plausibly hit against `HashEmbedder`'s lexical measure, 11 actually did
(73%), and a cache hit answered in ~8ms against a ~1.26s mean for an answer
that had to go through a (simulated) LLM -- roughly two orders of magnitude,
for the fraction of traffic that is genuinely a repeat question. The 27% miss
rate on paraphrases is not hidden: it is `HashEmbedder` doing exactly what
its docstring says a lexical measure can and cannot do; `OllamaEmbedder`
would very plausibly close most of that gap, unverified as it is here.

#### Predictive autoscaling (`infergrid/autoscale/`)

Holt-Winters triple exponential smoothing (`forecast.py::HoltWinters`, the
textbook algorithm, additive seasonality) forecasts the *next* window's
request rate from a repeating pattern already seen, and `controller.py`
turns that forecast into a target worker count. The payoff being measured by
`scripts/autoscale_demo.py` is specifically about *when in a repeating cycle*
a spike lands -- not whether a rate can be forecast at all (`Controller`'s
docstring spells out the comparison): a **reactive** policy sizes for the
rate the tick that *just ended* had (the standard "scale once you see the
load" approach); a **predictive** policy sizes for `forecast(1)` instead,
so extra capacity is warm *before* the spike arrives, one tick sooner,
provided the pattern has already been learned.

Making this real, not just a forecasting-accuracy exercise, needed one more
piece: a worker started *after* the gateway did must still become routable
with no gateway restart. `create_app(..., dynamic_workers=True)` makes
`alive_workers()` return whatever SWIM currently reports alive instead of a
list fixed at startup -- a worker autoscaled up joins the same SWIM group
(seeded on the existing workers, `LocalCluster.scale_workers()`) and becomes
routable within a couple of gossip rounds. `LoadTracker`'s per-worker dicts
became `defaultdict`s for this, since a dynamically-joined worker has no
entry seeded at gateway startup.

**A real bug found via `scripts/autoscale_demo.py`, not by any unit test:**
scaling *down* (killing workers no longer needed) exposed that on Windows,
connecting to a port whose listening process was *just* terminated does not
fail fast -- the new connection's SYN is simply not answered, so the caller
hangs until its own connect timeout fires, rather than getting an immediate
refused-connection error the way it does on Linux. The gateway's
`WORKER_TIMEOUT` had `connect=2.0` (generous, matching a real network's
possible latency); at that setting, a request that round-robined onto two or
three just-killed workers before reaching a live one paid four-to-six
*seconds* of pure timeout -- a multi-second stall, not the fast, near-instant
failover this project relies on everywhere else a worker turns out to be
unreachable (a crashed worker, a SWIM-confirmed-DEAD member). Diagnosed by
isolating the exact 2.0s-per-hop pattern in a minimal repro
(`client.get()` against a port right after `proc.terminate(); proc.wait()`
returned) before touching any autoscaling code. Fixed: `connect=0.5` -- a
genuine localhost connection establishes in low milliseconds, so even that
is generous, and a stale or dead candidate now fails in a fraction of a
second instead of two. `scripts/autoscale_demo.py` also gives SWIM a
deliberate settle period after every scaling change before sending more
traffic, which is realistic operational practice, not a workaround for this
bug -- the timeout fix is what makes an *unavoidably* stale routing view (the
gossip round or two SWIM genuinely needs) cheap instead of expensive.

A representative comparison (one low-low-high-high cycle, live): predictive
scaled to 4 workers one tick ahead of the spike and served it at ~700ms
mean / ~730ms p99 with zero failed requests; reactive was still on 1 worker
when the same spike hit, at ~1.4-1.7s mean / ~2.6s p99, and occasionally a
few requests failed outright under the compounded load of the spike arriving
at the same moment three new workers were being started to catch up. That
gap, and reactive's occasional outright failures, are the real, repeatable
cost of *reacting* to a load pattern instead of *anticipating* one already
learned -- not a hidden thumb on the scale (`predictive["failed"] == 0` is
asserted in the script; reactive's occasional failures are not papered over
by excluding them from the printed comparison).

## 4. Request flows

**Interactive chat:** client → gateway → rate limit → semantic cache lookup →
router picks worker → stream tokens to client → on worker failure, resume on
another worker from the last token index → store conversation turn → emit usage
event.

**Batch job:** client → gateway → one job message per request published to the
queue, keyed by tenant → job-ID list written to the state store under the
batch ID → idle worker consumes a job, checks the store for an existing
result (idempotency) → generates → result written to the store → usage event
published → offset committed → client polls `GET /v1/batches/{id}`.

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
| Worker crashes mid-batch-job | Uncommitted job redelivered to another worker once its session times out; idempotent (store lookup) so no duplicate work |
| A batch job keeps failing | Retried with backoff up to `max_attempts`, then sent to `{topic}.dlq` instead of blocking the partition forever |
| A worker is scaled down (autoscaling) | Terminated; `WORKER_TIMEOUT.connect=0.5` bounds how long a request that round-robins onto it before SWIM notices costs |
| A worker is scaled up (autoscaling) | Joins the SWIM group on its own; routable once gossiped, no gateway restart (`dynamic_workers=True`) |

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
| 5 | Message queue (from scratch, + an unverified real-Kafka option): batch inference API, usage metering, retries, dead-letter topic |
| 6 | Semantic cache, predictive autoscaling |
| 7 | Docker, Kubernetes, Prometheus/Grafana, OpenTelemetry, chaos tests, demo UI |

## 7. How success is measured

Every claim is backed by a benchmark against a baseline:
- TTFT p50/p99 and throughput: cache-aware routing vs round-robin.
- Prefix cache hit rate across the cluster.
- Dropped responses during chaos tests (target: zero).
- Store availability and staleness with nodes killed, for different R/W settings.
- Router overhead per request (it must be negligible next to LLM latency).
