"""Run the gateway: python -m infergrid.gateway --workers http://127.0.0.1:8701,http://127.0.0.1:8702"""

import argparse

import httpx
import uvicorn

from infergrid.gateway.app import create_app
from infergrid.gateway.rate_limit import RateLimiter
from infergrid.gateway.router import ROUTERS, make_router
from infergrid.membership import SwimNode
from infergrid.queue.cli import add_broker_args, build_queue_client
from infergrid.semantic_cache import HashEmbedder, OllamaEmbedder, SemanticCache
from infergrid.store.client import StoreClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the InferGrid gateway.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8700)
    parser.add_argument("--workers", required=True, help="comma-separated worker base URLs")
    parser.add_argument("--router", choices=ROUTERS, default="consistent_hash")
    parser.add_argument("--epsilon", type=float, default=0.25,
                        help="consistent_hash load bound: 0.25 lets a worker take 25%% above average; inf disables it")
    parser.add_argument("--max-failovers", type=int, default=2,
                        help="how many times one answer may move to another worker mid-stream (0 disables)")
    parser.add_argument("--swim-port", type=int, default=None,
                        help="UDP port for SWIM failure detection; omit to route to every configured worker blindly")
    parser.add_argument("--seeds", default="", help="comma-separated host:swim_port of workers to bootstrap from")
    parser.add_argument("--rate-limit-capacity", type=float, default=None,
                        help="per-tenant token bucket size; omit to disable rate limiting")
    parser.add_argument("--rate-limit-per-second", type=float, default=5.0, help="per-tenant refill rate")
    parser.add_argument("--hedge-delay-ms", type=float, default=None,
                        help="also try the next-best worker if no token arrives within this long; omit to disable")
    add_broker_args(parser)
    parser.add_argument("--store-url", default=None,
                        help="a state store node's base URL; needed for /v1/batches and --semantic-cache")
    parser.add_argument("--semantic-cache", action="store_true",
                        help="skip the LLM for a prompt within --semantic-cache-threshold of one already answered "
                             "(needs --store-url)")
    parser.add_argument("--semantic-cache-threshold", type=float, default=0.92)
    parser.add_argument("--semantic-cache-max-entries", type=int, default=200, help="per tenant")
    parser.add_argument("--embedder", choices=["hash", "ollama"], default="hash",
                        help="'hash' is a dependency-free bag-of-words stand-in; 'ollama' uses a real embedding "
                             "model via --embed-model, e.g. all-minilm (unverified in this project's own dev "
                             "environment, see infergrid/semantic_cache.py)")
    parser.add_argument("--embed-model", default="all-minilm")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--dynamic-workers", action="store_true",
                        help="route to whatever SWIM currently reports alive instead of a fixed --workers list, "
                             "so a worker started later (e.g. autoscaled up) is picked up with no restart; "
                             "needs --swim-port")
    args = parser.parse_args()

    workers = [w.strip() for w in args.workers.split(",") if w.strip()]
    router = make_router(args.router, args.epsilon)

    membership = None
    if args.swim_port:
        seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
        membership = SwimNode(f"{args.host}:{args.swim_port}", seeds=seeds)

    rate_limit = None
    if args.rate_limit_capacity:
        rate_limit = RateLimiter(args.rate_limit_capacity, args.rate_limit_per_second)

    queue_client = store_client = semantic_cache = None
    if args.store_url:
        side_client = httpx.AsyncClient()
        store_client = StoreClient(side_client, args.store_url)
        if args.queue_url or args.broker == "redpanda":
            queue_client = build_queue_client(args, side_client)
        if args.semantic_cache:
            embedder = (OllamaEmbedder(side_client, args.ollama_url, args.embed_model) if args.embedder == "ollama"
                       else HashEmbedder())
            semantic_cache = SemanticCache(store_client, embedder, args.semantic_cache_threshold,
                                           args.semantic_cache_max_entries)

    print(f"gateway on http://{args.host}:{args.port} -> {len(workers)} workers"
          + (" (+ whatever SWIM adds dynamically)" if args.dynamic_workers else "") + f", router={args.router}"
          + (f", swim on {args.host}:{args.swim_port}" if membership else "")
          + (f", rate limit {args.rate_limit_capacity}/{args.rate_limit_per_second}s" if rate_limit else "")
          + (f", hedge after {args.hedge_delay_ms}ms" if args.hedge_delay_ms else "")
          + (f", batches via {args.broker} broker" if queue_client else "")
          + (f", semantic cache ({args.embedder} embedder, threshold={args.semantic_cache_threshold})"
             if semantic_cache else ""))
    app = create_app(workers, router, max_failovers=args.max_failovers, membership=membership,
                     rate_limit=rate_limit, hedge_delay_ms=args.hedge_delay_ms,
                     queue_client=queue_client, store_client=store_client, semantic_cache=semantic_cache,
                     dynamic_workers=args.dynamic_workers)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
