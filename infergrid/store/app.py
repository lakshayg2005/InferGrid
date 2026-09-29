"""Store node HTTP API.

`/kv/{key}` is the client-facing coordinator API: any node accepts any key's read
or write and fans it out to that key's replicas per `StoreNode`. `/internal/kv`
is replica-to-replica: a direct replicated write/read, or -- when the body's
`hint_for` is set -- a hinted write this node is holding on another node's behalf
while that node is down (see node.py's module docstring).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from infergrid.store.node import Entry, StoreNode


class PutBody(BaseModel):
    value: Any


class InternalEntryBody(BaseModel):
    value: Any
    version: tuple[int, int, str]
    deleted: bool = False
    hint_for: str | None = None


def create_app(node: StoreNode) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await node.start()
        yield
        await node.stop()

    app = FastAPI(title=f"InferGrid store {node.addr}", lifespan=lifespan)

    @app.put("/kv/{key}")
    async def put_kv(key: str, body: PutBody, w: int | None = None) -> dict:
        ok = await node.put(key, body.value, w=w)
        if not ok:
            raise HTTPException(503, "write quorum not reached")
        return {"ok": True}

    @app.get("/kv/{key}")
    async def get_kv(key: str, r: int | None = None) -> dict:
        value = await node.get(key, r=r)
        if value is None:
            raise HTTPException(404, "not found")
        return {"value": value}

    @app.delete("/kv/{key}")
    async def delete_kv(key: str, w: int | None = None) -> dict:
        ok = await node.delete(key, w=w)
        if not ok:
            raise HTTPException(503, "write quorum not reached")
        return {"ok": True}

    @app.get("/internal/kv/{key}")
    async def internal_get(key: str) -> dict:
        entry = node.receive_get(key)
        if entry is None:
            raise HTTPException(404, "not found")
        return {"value": entry.value, "version": list(entry.version), "deleted": entry.deleted}

    @app.put("/internal/kv/{key}")
    async def internal_put(key: str, body: InternalEntryBody) -> dict:
        node.receive(key, Entry(body.value, body.version, body.deleted), body.hint_for)
        return {"ok": True}

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "addr": node.addr}

    @app.get("/stats")
    async def stats() -> dict:
        return {"addr": node.addr, "alive_nodes": sorted(node.alive_nodes()), **node.stats()}

    @app.get("/membership")
    async def membership() -> dict:
        return node.membership.snapshot() if node.membership else {}

    return app
