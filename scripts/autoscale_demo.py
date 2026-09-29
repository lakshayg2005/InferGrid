"""Real multi-process demo of Phase 6 predictive autoscaling (DESIGN.md
section 3.9): two separate gateway+worker clusters run the identical
synthetic low-low-high-high request-rate cycle, one scaled *predictively*
(Holt-Winters forecasts the next tick's rate and provisions for it ahead of
time) and one scaled *reactively* (sized only for the rate the tick that just
ended actually had, the standard "scale once you see the load" approach).
Both start from the same pre-seeded model of the pattern -- the payoff being
measured is forecasting *when in the cycle* a spike lands, not whether a
model can be learned at all (tests/test_forecast.py already covers that).

    python scripts/autoscale_demo.py                # ~60 seconds

New worker processes take real time to start and pass their health check;
predictive scaling pays that cost *before* the spike tick's requests are
sent, reactive pays it *during* -- which is exactly the latency gap this
script measures.
"""

import argparse
import asyncio
import statistics
import time

import httpx

from infergrid.autoscale.controller import Controller
from infergrid.autoscale.forecast import HoltWinters
from infergrid.local_cluster import LocalCluster

SEASON = 4  # low, low, high, high
PATTERN = [4, 4, 16, 16]  # requests per tick


async def send_tick(client: httpx.AsyncClient, gateway_url: str, n: int, max_tokens: int) -> tuple[list[float], int]:
    """(latencies of the ones that succeeded, count that failed). A failure is
    recorded, not raised: this script measures how a spike behaves under each
    scaling policy, which includes the policy occasionally not keeping up,
    not just the latency of the requests that got through."""
    async def one() -> float | None:
        started = time.perf_counter()
        try:
            resp = await client.post(f"{gateway_url}/v1/chat/completions",
                                     json={"messages": [{"role": "user", "content": "how do I reset my password?"}],
                                           "max_tokens": max_tokens})
            resp.raise_for_status()
        except httpx.HTTPError:
            return None
        return time.perf_counter() - started

    results = await asyncio.gather(*(one() for _ in range(n)))
    latencies = [r for r in results if r is not None]
    return latencies, len(results) - len(latencies)


async def run_pass(label: str, predictive: bool, args) -> dict:
    controller = Controller(HoltWinters(season_length=SEASON, alpha=0.6, beta=0.2, gamma=0.6),
                            requests_per_worker=args.requests_per_worker, min_workers=1, max_workers=args.max_workers,
                            headroom=1.2)
    controller.forecaster.seed([float(r) for r in PATTERN] * 4)  # 4 cycles of history, no live traffic needed

    print(f"\n--- {label} ---")
    with LocalCluster(workers=1, router="round_robin", dynamic_workers=True, gateway_port=args.base_port,
                      worker_base_port=args.base_port + 1, quiet=True) as cluster:
        async with httpx.AsyncClient(timeout=60.0) as client:
            tick_latencies: list[list[float]] = []
            worker_counts: list[int] = []
            total_failed = 0
            last_rate = float(PATTERN[-1])  # the pattern's last synthetic tick, continuing the same cycle live
            for tick_index in range(args.live_ticks):
                if predictive:
                    # seed() already absorbed every synthetic observation, so the model's
                    # current state already forecasts the very next (this) tick -- no
                    # update() here, or the forecast would skip ahead by one tick. The
                    # matching update(), with what this tick actually turns out to be,
                    # happens after it runs, preparing the forecast for the *next* one.
                    target = controller.target_workers(controller.forecaster.forecast(1))
                else:
                    target = controller.target_workers(last_rate)
                await asyncio.to_thread(cluster.scale_workers, target)
                # Realistic operational practice, not a fudge: give SWIM a moment to
                # converge on the new worker set before routing more traffic at it.
                # Skipping this is its own finding, not simulated here -- see
                # WORKER_TIMEOUT's comment in gateway/app.py for what a stale routing
                # view costs on Windows specifically (a multi-second stall per hop
                # onto a just-killed worker, not the fast failover this relies on
                # everywhere else) before that timeout was tightened.
                await asyncio.sleep(args.settle_seconds)
                worker_counts.append(target)

                n = PATTERN[tick_index % SEASON]
                latencies, failed = await send_tick(client, cluster.gateway_url, n, args.max_tokens)
                tick_latencies.append(latencies)
                total_failed += failed
                last_rate = float(n)
                if predictive:
                    controller.forecaster.update(last_rate)
                phase = "high" if n == max(PATTERN) else "low"
                summary = (f"mean={statistics.mean(latencies) * 1000:5.0f}ms p99={max(latencies) * 1000:5.0f}ms"
                          if latencies else "no requests succeeded")
                print(f"  tick {tick_index}: phase={phase:4s} n={n:2d} workers={target} {summary}"
                      + (f"  ({failed} FAILED)" if failed else ""))

    high_tick_latencies = [lat for i, latencies in enumerate(tick_latencies) if PATTERN[i % SEASON] == max(PATTERN)
                           for lat in latencies]
    return {"worker_counts": worker_counts, "failed": total_failed,
            "high_tick_mean_ms": statistics.mean(high_tick_latencies) * 1000 if high_tick_latencies else float("inf"),
            "high_tick_p99_ms": max(high_tick_latencies) * 1000 if high_tick_latencies else float("inf")}


async def run(args) -> bool:
    reactive = await run_pass("reactive (scales for the rate the tick that just ended had)", predictive=False,
                              args=args)
    predictive = await run_pass("predictive (Holt-Winters forecasts the next tick, scales ahead of it)",
                                predictive=True, args=args)

    print(f"\nspike-tick latency: reactive mean={reactive['high_tick_mean_ms']:.0f}ms "
          f"p99={reactive['high_tick_p99_ms']:.0f}ms ({reactive['failed']} failed requests overall) vs "
          f"predictive mean={predictive['high_tick_mean_ms']:.0f}ms p99={predictive['high_tick_p99_ms']:.0f}ms "
          f"({predictive['failed']} failed requests overall)")

    ok = (predictive["failed"] == 0 and predictive["high_tick_p99_ms"] < reactive["high_tick_p99_ms"])
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare predictive vs reactive autoscaling on a spiky workload.")
    parser.add_argument("--requests-per-worker", type=float, default=5.0,
                        help="provisioning assumption: requests per tick one worker can handle")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--live-ticks", type=int, default=4, help="SEASON=4, so 4 is one full low-low-high-high cycle")
    parser.add_argument("--settle-seconds", type=float, default=1.0,
                        help="pause after each scaling change for SWIM to converge before sending that tick's load")
    parser.add_argument("--base-port", type=int, default=8700)
    args = parser.parse_args()
    ok = asyncio.run(run(args))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
