"""Chaos test: kill workers under load and check that no answer is lost or corrupted.

    python scripts/chaos.py                    # with and without failover, ~3 minutes
    python scripts/chaos.py --kill-every 5     # harsher

While the chat workload runs, a random worker is killed abruptly (like a machine
losing power) every few seconds and restarted, with an empty cache, a few seconds
later. The same workload and the same kill schedule run twice: once with mid-stream
failover disabled and once enabled.

Every completed answer is compared word for word with the answer the simulator
gives when nothing fails, so "survived" means complete and correct, not just
"something came back".
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import httpx

from infergrid.common.schemas import ChatMessage
from infergrid.loadgen import Record, build_workload, percentile, run_workload
from infergrid.local_cluster import LocalCluster
from infergrid.worker.backends.sim import reference_answer

RESULTS_DIR = Path(__file__).resolve().parent.parent / "bench" / "results"


async def chaos_monkey(cluster: LocalCluster, schedule: list[tuple[float, int]], downtime: float,
                       started: float, log: list[str]) -> None:
    for at, index in schedule:
        await asyncio.sleep(max(0.0, started + at - time.perf_counter()))
        await asyncio.to_thread(cluster.kill_worker, index)
        log.append(f"t={at:5.1f}s killed worker-{index + 1}")
        await asyncio.sleep(downtime)
        await asyncio.to_thread(cluster.restart_worker, index)


async def run(cluster: LocalCluster, args, schedule) -> tuple[list[Record], list[Record], list[str], dict]:
    workload = build_workload(args.seed, args.duration, args.rate, apps=3)
    wrong: list[Record] = []

    def check(rec: Record, messages: list[dict], reply: str) -> None:
        pieces, _ = reference_answer([ChatMessage(**m) for m in messages], args.max_tokens)
        if reply != "".join(pieces):
            wrong.append(rec)

    log: list[str] = []
    started = time.perf_counter()
    monkey = asyncio.create_task(chaos_monkey(cluster, schedule, args.downtime, started, log))
    records, _ = await run_workload(cluster.gateway_url, workload, args.max_tokens, check)
    await monkey
    async with httpx.AsyncClient() as client:
        stats = (await client.get(f"{cluster.gateway_url}/stats")).json()
    return records, wrong, log, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Kill workers under load and count lost answers.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--rate", type=float, default=1.5, help="new conversations per second")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--kill-every", type=float, default=8.0, help="seconds between kills")
    parser.add_argument("--downtime", type=float, default=3.0, help="seconds before a killed worker restarts")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--base-port", type=int, default=8800)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    kill_times = [args.kill_every * k for k in range(1, int(args.duration / args.kill_every) + 1)]
    schedule = [(t, rng.randrange(args.workers)) for t in kill_times]

    rows, raw = [], []
    for i, max_failovers in enumerate([0, 2]):
        label = "failover off" if max_failovers == 0 else "failover on"
        port = args.base_port + 20 * i
        print(f"running with {label} ...", flush=True)
        with LocalCluster(workers=args.workers, router="consistent_hash", max_failovers=max_failovers,
                          gateway_port=port, worker_base_port=port + 1, quiet=True) as cluster:
            records, wrong, log, stats = asyncio.run(run(cluster, args, schedule))
        failed = [r for r in records if not r.ok]
        ttft = [r.ttft * 1000 for r in records if r.ok]
        rows.append((label, len(records), len(failed), len(wrong), stats["failovers"], percentile(ttft, 50),
                     percentile(ttft, 99)))
        print(f"  {len(log)} workers killed; {len(failed)} of {len(records)} answers lost, "
              f"{len(wrong)} corrupted, {stats['failovers']} mid-stream failovers; "
              f"in flight after the run: {sum(stats['in_flight'].values())}")
        raw.append({"run": label, "kills": log, "gateway_stats": stats, "records": [asdict(r) for r in records]})
        for r in failed[:3]:
            print(f"    e.g. conversation {r.conversation} turn {r.turn}: {r.error[:100]}")

    print()
    print("| Run | Requests | Answers lost | Answers corrupted | Mid-stream failovers | TTFT p50 | TTFT p99 |")
    print("|---|---|---|---|---|---|---|")
    for label, n, failed, wrong, failovers, p50, p99 in rows:
        print(f"| {label} | {n} | {failed} ({failed / n:.1%}) | {wrong} | {failovers} | {p50:.0f} ms | {p99:.0f} ms |")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"chaos-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"config": vars(args), "runs": raw}), encoding="utf-8")
    print(f"\nSaved {out.relative_to(RESULTS_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
