"""Run the gateway: python -m infergrid.gateway --workers http://127.0.0.1:8701,http://127.0.0.1:8702"""

import argparse

import uvicorn

from infergrid.gateway.app import create_app
from infergrid.gateway.router import ROUTERS, make_router


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the InferGrid gateway.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8700)
    parser.add_argument("--workers", required=True, help="comma-separated worker base URLs")
    parser.add_argument("--router", choices=ROUTERS, default="consistent_hash")
    parser.add_argument("--epsilon", type=float, default=0.25,
                        help="consistent_hash load bound: 0.25 lets a worker take 25%% above average; inf disables it")
    args = parser.parse_args()

    workers = [w.strip() for w in args.workers.split(",") if w.strip()]
    router = make_router(args.router, args.epsilon)
    print(f"gateway on http://{args.host}:{args.port} -> {len(workers)} workers, router={args.router}")
    uvicorn.run(create_app(workers, router), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
