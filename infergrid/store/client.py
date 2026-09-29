"""HTTP client for the Phase 4 state store's `/kv` API, for processes that use the
store as a client (the gateway's batch endpoints, a worker's batch consumer, the
billing consumer) without running a `StoreNode` themselves. Any one node can
coordinate any key (see store/node.py), so this only ever needs a single URL.
"""

from __future__ import annotations

from typing import Any

import httpx


class StoreClient:
    def __init__(self, http_client: httpx.AsyncClient, base_url: str, timeout: float = 5.0):
        self.http_client = http_client
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def get(self, key: str) -> Any | None:
        resp = await self.http_client.get(f"{self.base_url}/kv/{key}", timeout=self.timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()["value"]

    async def put(self, key: str, value: Any) -> None:
        resp = await self.http_client.put(f"{self.base_url}/kv/{key}", json={"value": value}, timeout=self.timeout)
        resp.raise_for_status()
