"""Run a worker: python -m infergrid.worker --id worker-1 --port 8701 --backend sim"""

import argparse

import httpx
import uvicorn

from infergrid.membership import SwimNode
from infergrid.queue.cli import add_broker_args, build_queue_client
from infergrid.store.client import StoreClient
from infergrid.worker.app import create_app
from infergrid.worker.backends import Backend, OllamaBackend, SimBackend, SimConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an InferGrid worker.")
    parser.add_argument("--id", default="worker-1", help="worker name shown in stats and logs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8701)
    parser.add_argument("--backend", choices=["sim", "ollama"], default="sim")
    parser.add_argument("--model", default="qwen2.5:0.5b", help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--max-queue", type=int, default=None,
                        help="requests refused once active+waiting reaches this (default: max_concurrency * 3)")
    parser.add_argument("--swim-port", type=int, default=None,
                        help="UDP port for SWIM failure detection; omit to disable membership")
    parser.add_argument("--seeds", default="", help="comma-separated host:swim_port of other members to bootstrap from")
    add_broker_args(parser)
    parser.add_argument("--store-url", default=None, help="a state store node's base URL; omit to disable batch consumption")
    parser.add_argument("--idle-queue-depth", type=int, default=1,
                        help="only pull a batch job while backend.queue_depth() is below this")
    args = parser.parse_args()

    backend: Backend
    if args.backend == "sim":
        config = SimConfig()
        if args.max_concurrency:
            config.max_concurrency = args.max_concurrency
        if args.max_queue:
            config.max_queue = args.max_queue
        backend = SimBackend(config)
    else:
        backend = OllamaBackend(args.model, args.ollama_url, args.max_concurrency or 2, args.max_queue)

    membership = None
    if args.swim_port:
        seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
        membership = SwimNode(f"{args.host}:{args.swim_port}", seeds=seeds,
                              metadata={"http_url": f"http://{args.host}:{args.port}",
                                        "capacity": backend.max_concurrency})

    queue_client = store_client = None
    if args.store_url and (args.queue_url or args.broker == "redpanda"):
        queue_store_client = httpx.AsyncClient()
        queue_client = build_queue_client(args, queue_store_client)
        store_client = StoreClient(queue_store_client, args.store_url)

    print(f"{args.id}: {args.backend} backend on http://{args.host}:{args.port}"
          + (f", swim on {args.host}:{args.swim_port}" if membership else "")
          + (f", batch jobs from {args.broker} broker" if queue_client else ""))
    app = create_app(backend, args.id, membership, queue_client=queue_client, store_client=store_client,
                     idle_queue_depth=args.idle_queue_depth)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
