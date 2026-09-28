"""Compare routing policies on the same simulated chat workload.

    python scripts/benchmark.py                              # all default policies, ~6 minutes
    python scripts/benchmark.py --repeats 3                  # 3 different workloads, ~18 minutes
    python scripts/benchmark.py --policies round_robin,consistent_hash:0.25 --duration 30

Each policy gets a fresh cluster (empty caches) and the exact same workload from
infergrid.loadgen: multi-turn chats sharing long system prompts, open-loop arrivals.

Simulated workers are deterministic, so every policy sees identical replies and
therefore identical prompts: the only thing that differs is the routing. SWIM
membership (Phase 3) is deliberately left off here -- its background gossip
traffic is a real but unrelated cost that would otherwise confound a benchmark
whose whole point is isolating routing-policy quality; scripts/chaos.py is
where membership's own effect is measured.
"""

import argparse
import asyncio
import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from infergrid.loadgen import Record, build_workload, percentile, run_workload
from infergrid.local_cluster import LocalCluster

RESULTS_DIR = Path(__file__).resolve().parent.parent / "bench" / "results"
DEFAULT_POLICIES = "round_robin,least_loaded,consistent_hash:inf,consistent_hash:0.25"


@dataclass
class PolicyResult:
    policy: str
    records: list[Record] = field(default_factory=list)
    wall_time: float = 0.0


def summarize(result: PolicyResult) -> dict:
    ok = [r for r in result.records if r.ok]
    ttft_ms = [r.ttft * 1000 for r in ok]
    per_worker: dict[str, int] = {}
    for r in ok:
        per_worker[r.worker] = per_worker.get(r.worker, 0) + 1
    prompt = sum(r.prompt_tokens for r in ok)
    return {
        "policy": result.policy,
        "requests": len(result.records),
        "errors": len(result.records) - len(ok),
        "ttft_p50_ms": percentile(ttft_ms, 50),
        "ttft_p95_ms": percentile(ttft_ms, 95),
        "ttft_p99_ms": percentile(ttft_ms, 99),
        "ttft_mean_ms": statistics.fmean(ttft_ms) if ttft_ms else float("nan"),
        "cache_hit_rate": sum(r.cached_tokens for r in ok) / prompt if prompt else 0.0,
        "prefill_tokens": prompt - sum(r.cached_tokens for r in ok),
        "load_imbalance": max(per_worker.values()) / statistics.fmean(per_worker.values()) if per_worker else 0.0,
        "requests_per_worker": dict(sorted(per_worker.items())),
        "throughput_rps": len(ok) / result.wall_time if result.wall_time else 0.0,
    }


def parse_policy(spec: str) -> tuple[str, float]:
    name, _, epsilon = spec.partition(":")
    return name, float(epsilon) if epsilon else 0.25


def print_table(runs: dict[str, list[dict]]) -> None:
    """One row per policy: the mean across repeats, with the (min-max) range for latencies."""

    def spread(summaries: list[dict], key: str) -> str:
        values = [s[key] for s in summaries]
        text = f"{statistics.fmean(values):.0f} ms"
        return text + f" ({min(values):.0f}-{max(values):.0f})" if len(values) > 1 else text

    def mean(summaries: list[dict], key: str) -> float:
        return statistics.fmean(s[key] for s in summaries)

    header = ("| Policy | TTFT p50 | TTFT p95 | TTFT p99 | Cache hit rate | Tokens prefilled | "
              "Busiest worker vs avg | Errors |")
    print(header)
    print("|---" * (header.count("|") - 1) + "|")
    for policy, summaries in runs.items():
        print(f"| {policy} | {spread(summaries, 'ttft_p50_ms')} | {spread(summaries, 'ttft_p95_ms')} | "
              f"{spread(summaries, 'ttft_p99_ms')} | {mean(summaries, 'cache_hit_rate'):.1%} | "
              f"{mean(summaries, 'prefill_tokens'):,.0f} | {mean(summaries, 'load_imbalance'):.2f}x | "
              f"{sum(s['errors'] for s in summaries)} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark InferGrid routing policies.")
    parser.add_argument("--policies", default=DEFAULT_POLICIES,
                        help="comma-separated; consistent_hash takes an epsilon, e.g. consistent_hash:0.25 or :inf")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--duration", type=float, default=60.0, help="seconds during which conversations start")
    parser.add_argument("--rate", type=float, default=1.5, help="new conversations per second")
    parser.add_argument("--apps", type=int, default=3, help="distinct system prompts")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=1,
                        help="run every policy on this many different workloads (seeds seed, seed+1, ...)")
    parser.add_argument("--base-port", type=int, default=8750)
    args = parser.parse_args()

    policies = [parse_policy(spec) for spec in args.policies.split(",")]
    runs: dict[str, list[dict]] = {}
    raw = []
    port = args.base_port
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        workload = build_workload(seed, args.duration, args.rate, args.apps)
        turns = sum(len(c.user_messages) for c in workload)
        print(f"Workload seed {seed}: {len(workload)} conversations, {turns} requests, {args.workers} workers")

        # Policies take turns within each repeat, so slow drift in the machine's
        # background load is spread across all of them rather than hitting one.
        for name, epsilon in policies:
            label = name if name != "consistent_hash" else f"consistent_hash (eps={epsilon:g})"
            port += 20  # fresh ports per run, so a slow shutdown never collides
            with LocalCluster(workers=args.workers, router=name, epsilon=epsilon, membership=False,
                              gateway_port=port, worker_base_port=port + 1, quiet=True) as cluster:
                print(f"  running {label} ...", flush=True)
                records, wall = asyncio.run(run_workload(cluster.gateway_url, workload, args.max_tokens))
            summary = summarize(PolicyResult(label, records, wall))
            runs.setdefault(label, []).append(summary)
            raw.append({"policy": label, "seed": seed, "records": [asdict(r) for r in records]})
            print(f"    TTFT p50 {summary['ttft_p50_ms']:.0f} ms, p99 {summary['ttft_p99_ms']:.0f} ms, "
                  f"cache hit rate {summary['cache_hit_rate']:.1%}")

    print()
    print_table(runs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"routing-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"config": vars(args), "results": runs, "raw": raw}), encoding="utf-8")
    print(f"\nSaved {out.relative_to(RESULTS_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
