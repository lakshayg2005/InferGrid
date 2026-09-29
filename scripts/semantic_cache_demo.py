"""Real multi-process demo of the Phase 6 semantic cache: a gateway, a
single-node store and two simulated workers, fed a workload of FAQ-style
questions asked repeatedly with light paraphrasing, interspersed with unique
ones. Measures the actual latency gap between a cache miss (goes to a worker,
pays the simulated model's real decode time) and a cache hit (answered from
the store, no worker involved at all) -- the concrete payoff DESIGN.md
section 3.9 claims, not just that the logic works in isolation
(tests/test_semantic_cache.py and tests/test_semantic_cache_gateway.py
already cover that).

    python scripts/semantic_cache_demo.py                # ~15 seconds
"""

import argparse
import asyncio
import random
import statistics
import time

import httpx

from infergrid.local_cluster import LocalCluster

FAQ = [
    "How do I reset my password?",
    "What is your refund policy?",
    "How do I cancel my subscription?",
    "Where can I download my invoice?",
    "How do I change my billing email?",
]

PARAPHRASES = {
    # HashEmbedder measures word overlap, not meaning (see semantic_cache.py's
    # docstring), so these stay close to the original wording -- what a
    # bag-of-words hash can actually be expected to catch, not what a real
    # embedding model could. A couple are deliberately looser and expected to
    # miss; the script reports the real resulting hit rate rather than assuming
    # every variant hits.
    "How do I reset my password?": ["how do i reset my password", "How can I reset my password?",
                                    "How do I reset my account password?"],
    "What is your refund policy?": ["what is your refund policy", "What's your refund policy?",
                                    "What is your policy on refunds?"],
    "How do I cancel my subscription?": ["how do i cancel my subscription", "How can I cancel my subscription?",
                                         "How do I cancel my paid subscription?"],
    "Where can I download my invoice?": ["where can i download my invoice", "Where can I download my invoice PDF?",
                                         "How do I download my invoice?"],
    "How do I change my billing email?": ["how do i change my billing email", "How can I change my billing email?",
                                          "How do I update my billing email address?"],
}

# Deliberately different vocabulary from the FAQ and from each other: a
# bag-of-words embedder scores any two texts sharing most of their words as
# similar regardless of topic, so a templated "question number N" set here
# would falsely match against itself.
UNIQUE = [
    "What time does the office open on weekends?",
    "Can I upgrade my plan in the middle of a billing cycle?",
    "Do you support two-factor authentication?",
    "Is there a mobile app available for iOS?",
    "What payment methods do you accept?",
    "How long does shipping usually take?",
    "Is there a discount for paying annually?",
    "Do you offer customer support over the phone?",
    "What happens if I exceed my monthly usage limit?",
    "Can I export my data as a CSV file?",
]


async def ask(client: httpx.AsyncClient, gateway_url: str, text: str) -> tuple[float, str, str]:
    started = time.perf_counter()
    resp = await client.post(f"{gateway_url}/v1/chat/completions",
                             json={"messages": [{"role": "user", "content": text}], "max_tokens": 40},
                             headers={"x-tenant-id": "acme"})
    resp.raise_for_status()
    elapsed = time.perf_counter() - started
    return elapsed, resp.headers["x-infergrid-worker"], resp.json()["choices"][0]["message"]["content"]


async def run(cluster: LocalCluster, args) -> bool:
    rng = random.Random(args.seed)
    # Each FAQ's canonical form always comes before its own paraphrases (a
    # paraphrase can only ever hit something already cached); blocks are
    # shuffled against each other and against the unrelated questions, not
    # the individual questions within a block.
    blocks = [[canonical, *PARAPHRASES[canonical]] for canonical in FAQ] + [[q] for q in UNIQUE]
    rng.shuffle(blocks)
    workload = [text for block in blocks for text in block]

    async with httpx.AsyncClient(timeout=30.0) as client:
        misses, hits = [], []
        answers_by_canonical: dict[str, str] = {}
        wrong = []
        for text in workload:
            elapsed, worker, content = await ask(client, cluster.gateway_url, text)
            if worker == "semantic-cache":
                hits.append(elapsed)
                canonical = next((c for c, variants in PARAPHRASES.items() if text in variants), None)
                if canonical and canonical in answers_by_canonical and content != answers_by_canonical[canonical]:
                    wrong.append(text)
            else:
                misses.append(elapsed)
                if text in FAQ:
                    answers_by_canonical[text] = content
            await asyncio.sleep(0.02)  # let the fire-and-forget cache write land before the next ask

        stats = (await client.get(f"{cluster.gateway_url}/stats")).json()

    possible_hits = sum(len(v) for v in PARAPHRASES.values())
    print(f"{len(workload)} requests: {len(misses)} misses, {len(hits)} hits "
          f"(gateway /stats agrees: {stats['cache_misses']} misses, {stats['cache_hits']} hits)")
    print(f"of {possible_hits} paraphrases that could plausibly hit (HashEmbedder measures word "
          f"overlap, not meaning -- see semantic_cache.py's docstring), {len(hits)} actually did "
          f"({len(hits) / possible_hits:.0%})")
    if hits:
        print(f"mean miss latency {statistics.mean(misses) * 1000:.0f}ms vs "
              f"mean hit latency {statistics.mean(hits) * 1000:.0f}ms "
              f"({statistics.mean(misses) / statistics.mean(hits):.0f}x faster)")
    if wrong:
        print(f"WRONG (a hit didn't return the canonical answer): {wrong[:3]}")

    # Every UNIQUE question and every FAQ's first ask is a guaranteed miss (nothing to
    # hit yet); some paraphrases missing too is expected and reported above, not
    # asserted to be zero -- that would be grading the lexical embedder on a semantic
    # task it was never meant to pass.
    ok = (len(misses) >= len(FAQ) + len(UNIQUE) and len(hits) == stats["cache_hits"] and not wrong
          and len(hits) > 0 and statistics.mean(hits) < statistics.mean(misses) / 2)
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the semantic cache's real hit rate and latency payoff.")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="HashEmbedder is lexical, not semantic -- a looser threshold than the default 0.92 "
                             "is needed for it to recognize these paraphrases as the same question")
    parser.add_argument("--base-port", type=int, default=8700)
    args = parser.parse_args()

    with LocalCluster(workers=2, router="round_robin", semantic_cache=True, semantic_cache_threshold=args.threshold,
                      gateway_port=args.base_port, worker_base_port=args.base_port + 1,
                      store_port=args.base_port + 250, quiet=True) as cluster:
        ok = asyncio.run(run(cluster, args))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
