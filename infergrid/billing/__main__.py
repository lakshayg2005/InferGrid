"""Run the billing consumer: python -m infergrid.billing --queue-url http://127.0.0.1:8900 --store-url http://127.0.0.1:8801"""

import argparse
import asyncio
import functools

import httpx

from infergrid.billing.consumer import apply_usage_event
from infergrid.queue.cli import add_broker_args, build_queue_client
from infergrid.queue.consumer import consume_with_retry
from infergrid.store.client import StoreClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the InferGrid billing consumer.")
    add_broker_args(parser)
    parser.add_argument("--store-url", required=True)
    parser.add_argument("--consumer-id", default="billing-1")
    args = parser.parse_args()
    if args.broker == "infergrid" and not args.queue_url:
        parser.error("--queue-url is required unless --broker redpanda is used")

    async def run() -> None:
        async with httpx.AsyncClient() as http:
            queue = build_queue_client(args, http)
            store = StoreClient(http, args.store_url)
            print(f"billing consumer {args.consumer_id} ({args.broker} broker) -> {args.store_url}")
            await consume_with_retry(queue, "usage", "billing", args.consumer_id,
                                     functools.partial(apply_usage_event, store), dlq_topic="usage.dlq")

    asyncio.run(run())


if __name__ == "__main__":
    main()
