"""Start a local cluster: N workers plus a gateway, each in its own process.

    python scripts/run_local.py --workers 3                      # simulated LLMs
    python scripts/run_local.py --workers 2 --backend ollama     # real model via Ollama
    python scripts/run_local.py --router round_robin             # compare routing policies
    python scripts/run_local.py --no-membership                  # disable SWIM failure detection

Press Ctrl+C to stop everything.
"""

import argparse
import sys
import time

from infergrid.gateway.router import ROUTERS
from infergrid.local_cluster import LocalCluster


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an InferGrid cluster locally.")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--backend", choices=["sim", "ollama"], default="sim")
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--router", choices=ROUTERS, default="consistent_hash")
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--max-failovers", type=int, default=2)
    parser.add_argument("--no-membership", action="store_true", help="disable SWIM failure detection")
    parser.add_argument("--hedge-delay-ms", type=float, default=None)
    parser.add_argument("--gateway-port", type=int, default=8700)
    parser.add_argument("--worker-base-port", type=int, default=8701)
    args = parser.parse_args()

    cluster = LocalCluster(
        workers=args.workers, backend=args.backend, model=args.model, router=args.router, epsilon=args.epsilon,
        max_failovers=args.max_failovers, membership=not args.no_membership, hedge_delay_ms=args.hedge_delay_ms,
        gateway_port=args.gateway_port, worker_base_port=args.worker_base_port,
    )
    try:
        cluster.start()
    except RuntimeError as exc:
        sys.exit(str(exc))

    print(f"\nCluster ready. Gateway: {cluster.gateway_url}  router={args.router}  "
          f"membership={'on' if cluster.membership else 'off'}  (Ctrl+C to stop)\n")
    try:
        # Workers are allowed to die (that is what failover is for); the gateway is not.
        reported = set()
        while cluster.gateway_alive():
            for url, code in cluster.exited_workers():
                if url not in reported:
                    print(f"worker {url} exited with code {code}")
                    reported.add(url)
            time.sleep(0.5)
        print("gateway exited; shutting down")
    except KeyboardInterrupt:
        pass
    finally:
        cluster.stop()


if __name__ == "__main__":
    main()
