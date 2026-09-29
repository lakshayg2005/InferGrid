"""Run a broker: python -m infergrid.queue.broker --port 8900"""

import argparse

import uvicorn

from infergrid.queue.broker.app import create_app
from infergrid.queue.broker.core import DEFAULT_PARTITIONS, DEFAULT_SESSION_TIMEOUT, Broker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an InferGrid queue broker.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--partitions", type=int, default=DEFAULT_PARTITIONS)
    parser.add_argument("--session-timeout", type=float, default=DEFAULT_SESSION_TIMEOUT,
                        help="seconds a consumer may go quiet before its partitions are reassigned")
    args = parser.parse_args()

    broker = Broker(num_partitions=args.partitions, session_timeout=args.session_timeout)
    print(f"queue broker on http://{args.host}:{args.port}: {args.partitions} partitions per topic")
    uvicorn.run(create_app(broker), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
