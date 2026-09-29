"""Real multi-process demo of the Phase 5 batch pipeline: gateway, queue broker,
a single-node store, two workers and a billing consumer, all as separate OS
processes. Submits an interactive request and several small batches spread
across different tenants (so they land on different queue partitions and, most
likely, different workers), kills one worker mid-processing, and checks that
every job still completes correctly -- some of them via at-least-once
redelivery to the surviving worker -- and that usage metering billed both the
interactive request and every batch job.

    python scripts/batch_demo.py                # ~40 seconds

See tests/test_batch.py for the fast in-process version of this same pipeline
(including retry/DLQ, which this script does not re-demonstrate) and
DESIGN.md section 3.6 for the design.
"""

import argparse
import asyncio
import time

import httpx

from infergrid.common.schemas import ChatMessage
from infergrid.local_cluster import LocalCluster
from infergrid.worker.backends.sim import reference_answer


async def usage_for(client: httpx.AsyncClient, store_url: str, tenant: str) -> int:
    resp = await client.get(f"{store_url}/kv/usage:{tenant}")
    return resp.json()["value"] if resp.status_code == 200 else 0


async def run(cluster: LocalCluster, args) -> bool:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{cluster.gateway_url}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 10},
            headers={"x-tenant-id": "acme"},
        )
        resp.raise_for_status()

        tenants = [f"acme-{i}" for i in range(args.tenants)]
        jobs_per_tenant, batch_ids = [], []
        for tenant in tenants:
            jobs = [{"messages": [{"role": "user", "content": f"{tenant} question {i}"}], "max_tokens": args.max_tokens}
                    for i in range(args.jobs_per_tenant)]
            jobs_per_tenant.append(jobs)
            resp = await client.post(f"{cluster.gateway_url}/v1/batches", json={"requests": jobs},
                                     headers={"x-tenant-id": tenant})
            resp.raise_for_status()
            batch_ids.append(resp.json()["batch_id"])
        total_jobs = sum(len(j) for j in jobs_per_tenant)
        print(f"submitted {len(tenants)} batches, {total_jobs} jobs total, across tenants {tenants}")

        await asyncio.sleep(args.kill_after)
        print("killing worker-1 mid-batch (no graceful shutdown) ...")
        await asyncio.to_thread(cluster.kill_worker, 0)

        deadline = time.monotonic() + args.wait
        statuses = []
        while time.monotonic() < deadline:
            statuses = [(await client.get(f"{cluster.gateway_url}/v1/batches/{bid}")).json() for bid in batch_ids]
            if all(s["done"] == s["total"] for s in statuses):
                break
            await asyncio.sleep(0.5)
        done = sum(s["done"] for s in statuses)
        print(f"{done}/{total_jobs} jobs done after up to {args.wait}s "
              f"(worker-1's stuck work needed the broker's {args.session_timeout_note}s session timeout to be reassigned)")
        if done < total_jobs:
            print("FAIL: not every job completed")
            return False

        wrong = []
        for jobs, status in zip(jobs_per_tenant, statuses):
            for job, result in zip(jobs, status["jobs"]):
                expected, _ = reference_answer([ChatMessage(**m) for m in job["messages"]], job["max_tokens"])
                if result["content"] != "".join(expected):
                    wrong.append(result["job_id"])
        print(f"{total_jobs - len(wrong)}/{total_jobs} batch results correct"
              + (f", WRONG: {wrong[:3]}" if wrong else ""))

        await asyncio.sleep(1.0)  # let the billing consumer catch up on the last usage events
        interactive_usage = await usage_for(client, cluster.store_url, "acme")
        batch_usage = sum([await usage_for(client, cluster.store_url, t) for t in tenants])
        print(f"billed usage: interactive={interactive_usage} tokens, batch={batch_usage} tokens "
              f"across {len(tenants)} tenants")

        ok = not wrong and done == total_jobs and interactive_usage > 0 and batch_usage > 0
        print(f"\n{'PASS' if ok else 'FAIL'}")
        return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit batches, kill a worker mid-processing, check recovery.")
    parser.add_argument("--tenants", type=int, default=4)
    parser.add_argument("--jobs-per-tenant", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--kill-after", type=float, default=0.3, help="seconds after submitting before killing worker-1")
    parser.add_argument("--wait", type=float, default=45.0, help="seconds to wait for every job to complete")
    parser.add_argument("--base-port", type=int, default=8700)
    args = parser.parse_args()
    args.session_timeout_note = 10  # matches infergrid.queue.broker's default --session-timeout

    with LocalCluster(workers=2, router="round_robin", queue=True, gateway_port=args.base_port,
                      worker_base_port=args.base_port + 1, queue_port=args.base_port + 200,
                      store_port=args.base_port + 250, quiet=True) as cluster:
        ok = asyncio.run(run(cluster, args))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
