"""HTTP transport between store nodes, over the `/internal/kv` endpoint in app.py."""

from __future__ import annotations

import httpx

from infergrid.store.node import Entry


class HttpTransport:
    def __init__(self, http_client: httpx.AsyncClient, timeout: float = 2.0):
        self.http_client = http_client
        self.timeout = timeout

    async def get(self, peer: str, key: str) -> Entry | None:
        resp = await self.http_client.get(f"{peer}/internal/kv/{key}", timeout=self.timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        body = resp.json()
        return Entry(body["value"], tuple(body["version"]), body["deleted"])

    async def put(self, peer: str, key: str, entry: Entry, hint_for: str | None = None) -> bool:
        resp = await self.http_client.put(
            f"{peer}/internal/kv/{key}",
            json={"value": entry.value, "version": list(entry.version), "deleted": entry.deleted, "hint_for": hint_for},
            timeout=self.timeout,
        )
        return resp.status_code == 200
