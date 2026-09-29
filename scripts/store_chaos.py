"""Chaos test for the Phase 4 state store: kill a real node process under write/read
load and check that sloppy quorum, hinted handoff and read repair actually do what
DESIGN.md section 3.5 claims, against a real multi-process cluster -- not just the
in-memory fake-transport used by tests/test_store.py.

    python scripts/store_chaos.py                # ~15 seconds

Sequence: write a batch of keys with the full cluster up, kill one node, write and
read another batch while it's down (this must still succeed via sloppy quorum,
and every read must still return the correct value), restart the node, wait for
SWIM to notice and the hint-retry loop to run, then check -- directly against
that node's own storage, bypassing coordination -- what actually got restored.

This deliberately separates two different things a crash can lose, because
hinted handoff only heals one of them: a write that happened *while the node was
down* got redirected to a substitute holding a hint, and that hint is delivered
once the node is back -- but data the node already held *before* it crashed is
gone from its (in-memory, unpersisted) storage and no hint exists for it, since
nothing wrote it again during the outage. That gap is only closed lazily, by
read repair, the next time something actually reads an affected key -- there is
no anti-entropy sweep in this implementation (DESIGN.md 3.5 lists Merkle-tree
anti-entropy as an unimplemented stretch goal). So: keys written *during* the
outage are checked for delivery by hinted handoff alone; keys the node already
held *before* the crash are expected to be gone from its own storage right after
restart, and are only checked via the normal coordinated read path, which must
still return the right answer from the surviving replicas regardless.
"""

import argparse
import asyncio
import random
import string
import time

import httpx

from infergrid.common.hashring import HashRing
from infergrid.local_cluster import StoreCluster


def random_value(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_lowercase, k=12))


async def write_batch(
    client: httpx.AsyncClient, urls: list[str], keys: list[str], rng: random.Random
) -> tuple[dict[str, str], list[str]]:
    """(succeeded, failed). A write that comes back non-200 is recorded, not raised:
    a chaos test's job is to count how often quorum fails under real failure and
    real load, not to crash the first time it does."""
    expected: dict[str, str] = {}
    failed: list[str] = []
    for key in keys:
        value = random_value(rng)
        coordinator = rng.choice(urls)
        resp = await client.put(f"{coordinator}/kv/{key}", json={"value": value})
        if resp.status_code == 200:
            expected[key] = value
        else:
            failed.append(f"{key}: {resp.status_code} via {coordinator}")
        await asyncio.sleep(0.05)
    return expected, failed


async def read_batch(client: httpx.AsyncClient, urls: list[str], expected: dict[str, str], rng: random.Random) -> list[str]:
    wrong = []
    for key, value in expected.items():
        coordinator = rng.choice(urls)
        resp = await client.get(f"{coordinator}/kv/{key}")
        got = resp.json()["value"] if resp.status_code == 200 else None
        if got != value:
            wrong.append(f"{key}: expected {value!r}, got {got!r} (via {coordinator})")
    return wrong


async def run(cluster: StoreCluster, args) -> None:
    rng = random.Random(args.seed)
    urls = cluster.node_urls
    async with httpx.AsyncClient(timeout=5.0) as client:
        before_keys = [f"before-{i}" for i in range(args.keys)]
        print(f"writing {len(before_keys)} keys with all {len(urls)} nodes up ...")
        before, failed_before = await write_batch(client, urls, before_keys, rng)
        wrong = await read_batch(client, urls, before, rng)
        print(f"  {len(before) - len(wrong)}/{len(before_keys)} correct"
              + (f", WRONG: {wrong[:3]}" if wrong else "") + (f", write FAILED: {failed_before[:3]}" if failed_before else ""))

        victim = rng.randrange(len(urls))
        print(f"killing node {victim} ({urls[victim]}) ...")
        await asyncio.to_thread(cluster.kill, victim)

        during_keys = [f"during-{i}" for i in range(args.keys)]
        other_urls = [u for i, u in enumerate(urls) if i != victim]
        print(f"writing {len(during_keys)} keys with node {victim} down (sloppy quorum) ...")
        during, failed_during = await write_batch(client, other_urls, during_keys, rng)
        wrong_during = await read_batch(client, other_urls, {**before, **during}, rng)
        print(f"  {len(before) + len(during) - len(wrong_during)}/{len(before_keys) + len(during_keys)} correct"
              + (f", WRONG: {wrong_during[:3]}" if wrong_during else "")
              + (f", write FAILED: {failed_during[:3]}" if failed_during else ""))

        print(f"restarting node {victim} ...")
        ok = await asyncio.to_thread(cluster.restart, victim)
        if not ok:
            print("  FAILED to restart")
            return

        print(f"restarted node {victim}; waiting for SWIM detection + hint delivery (up to {args.wait}s) ...")
        victim_url = urls[victim]
        ring = HashRing(urls, vnodes=100)
        # Only "during" keys the victim owns get a hint (see module docstring); "before"
        # keys it owned are gone from its own storage until something reads them again.
        owned_during = {k: v for k, v in during.items() if victim_url in ring.preference_list(k, args.n_replicas)}
        print(f"  node {victim} owns {len(owned_during)}/{len(during)} of the during-outage keys")

        deadline = time.monotonic() + args.wait
        delivered = False
        while time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            missing = 0
            for key, value in owned_during.items():
                resp = await client.get(f"{victim_url}/internal/kv/{key}")
                if resp.status_code != 200 or resp.json()["value"] != value:
                    missing += 1
            if missing == 0:
                delivered = True
                break
            print(f"  ...{len(owned_during) - missing}/{len(owned_during)} recovered by hinted handoff so far")
        print(f"hinted handoff delivered every during-outage key it owns: {delivered}")

        all_keys = {**before, **during}
        wrong_after = await read_batch(client, urls, all_keys, rng)
        print(f"coordinated read-back across all {len(urls)} nodes (survivors must mask the crash entirely): "
              f"{len(all_keys) - len(wrong_after)}/{len(all_keys)} correct"
              + (f", WRONG: {wrong_after[:3]}" if wrong_after else ""))

        await asyncio.sleep(1.0)  # let each coordinator's fire-and-forget read-repair task finish
        owned_before = {k: v for k, v in before.items() if victim_url in ring.preference_list(k, args.n_replicas)}
        healed = 0
        for k, v in owned_before.items():
            r = await client.get(f"{victim_url}/internal/kv/{k}")
            if r.status_code == 200 and r.json()["value"] == v:
                healed += 1
        print(f"read repair backfilled {healed}/{len(owned_before)} of the victim's pre-crash keys "
              f"as a side effect of the reads above")

        if wrong_after:
            print("\ndiagnosing WRONG reads (ring preference list + each node's alive_nodes view):")
            for k in [w.split(":")[0] for w in wrong_after]:
                pref = ring.preference_list(k, args.n_replicas)
                print(f"  {k}: preference_list={pref}")
                for url in urls:
                    stats = (await client.get(f"{url}/stats")).json()
                    print(f"    {url}: alive_nodes={stats['alive_nodes']}, keys={stats['keys']}")

        ok = not wrong and not wrong_during and not wrong_after and not failed_before and not failed_during and delivered
        print(f"\n{'PASS' if ok else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kill a store node under load and check quorum reads/writes.")
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--n-replicas", type=int, default=3)
    parser.add_argument("--w", type=int, default=2)
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--keys", type=int, default=30)
    parser.add_argument("--wait", type=float, default=15.0, help="seconds to wait for hinted handoff after restart")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--base-port", type=int, default=8801)
    parser.add_argument("--verbose", action="store_true",
                        help="show each node process's own stderr, including its quorum-failure diagnostics")
    args = parser.parse_args()

    with StoreCluster(nodes=args.nodes, n_replicas=args.n_replicas, w=args.w, r=args.r,
                       base_port=args.base_port, quiet=not args.verbose) as cluster:
        asyncio.run(run(cluster, args))


if __name__ == "__main__":
    main()
