"""Run a store node: python -m infergrid.store --port 8801 --nodes http://127.0.0.1:8801 ... """

import argparse

import httpx
import uvicorn

from infergrid.membership import SwimNode
from infergrid.store.app import create_app
from infergrid.store.node import StoreNode
from infergrid.store.transport import HttpTransport


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an InferGrid state store node.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8801)
    parser.add_argument("--nodes", nargs="+", required=True, help="http base URL of every node in the cluster, including this one")
    parser.add_argument("--n-replicas", type=int, default=3)
    parser.add_argument("--w", type=int, default=2)
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--swim-port", type=int, default=None,
                        help="UDP port for SWIM failure detection; omit to assume every node in --nodes is always up")
    parser.add_argument("--seeds", default="", help="comma-separated host:swim_port of other members to bootstrap from")
    args = parser.parse_args()

    addr = f"http://{args.host}:{args.port}"

    membership = None
    if args.swim_port:
        seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
        # Wider than worker/gateway's SWIM defaults (worker/__main__.py): every read
        # or write here fans out concurrent HTTP replication to 2-3 peers on the same
        # event loop that has to answer SWIM pings, so a coordinator busy replicating
        # is slower to service its own ping/ack than a worker just generating tokens.
        # Found via scripts/store_chaos.py: tighter timeouts produced real false
        # suspicion (and occasionally false death) of nodes that were never touched.
        membership = SwimNode(f"{args.host}:{args.swim_port}", seeds=seeds, metadata={"http_url": addr},
                              ping_timeout=1.0, suspicion_timeout=4.0)

    node = StoreNode(
        addr,
        args.nodes,
        HttpTransport(httpx.AsyncClient()),
        n_replicas=args.n_replicas,
        w=args.w,
        r=args.r,
        membership=membership,
    )
    print(f"store node {addr}: {len(args.nodes)} nodes, n={args.n_replicas} w={args.w} r={args.r}"
          + (f", swim on {args.host}:{args.swim_port}" if membership else ""))
    uvicorn.run(create_app(node), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
