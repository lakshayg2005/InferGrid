"""Shared --broker/--queue-url/--bootstrap-servers CLI wiring, used identically
by gateway, worker and billing __main__.py so each doesn't repeat it.
"""

from __future__ import annotations

import argparse

import httpx

from infergrid.queue.base import QueueClient


def add_broker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--queue-url", default=None, help="infergrid.queue.broker base URL (--broker infergrid)")
    parser.add_argument("--broker", choices=["infergrid", "redpanda"], default="infergrid",
                        help="'redpanda' talks to a real Kafka-API broker via --bootstrap-servers instead -- "
                             "unverified in this project's own environment, see infergrid/queue/redpanda.py")
    parser.add_argument("--bootstrap-servers", default="127.0.0.1:9092", help="for --broker redpanda")


def build_queue_client(args: argparse.Namespace, http_client: httpx.AsyncClient) -> QueueClient | None:
    if args.broker == "redpanda":
        from infergrid.queue.redpanda import RedpandaClient
        return RedpandaClient(args.bootstrap_servers)
    if args.queue_url:
        from infergrid.queue.client import BrokerClient
        return BrokerClient(http_client, args.queue_url)
    return None
