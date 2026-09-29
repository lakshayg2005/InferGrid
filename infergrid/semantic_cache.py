"""Semantic response cache (DESIGN.md section 3.9): a new prompt within a
cosine-similarity threshold of one already answered is served from the cache
instead of ever reaching a worker -- unlike the worker-side prefix cache
(`worker/prefix_cache.py`), which still recomputes the *new* tokens of a
similar-but-not-identical prompt, this skips generation entirely when the
question has, in substance, already been asked and answered.

Two `Embedder` implementations, the same split as `worker/backends/` (sim vs
Ollama) and `queue/` (from-scratch broker vs Redpanda):

- `HashEmbedder`: a deterministic, no-dependency stand-in -- the hashing
  trick (each word hashes into one of `dims` buckets, counted, L2-normalized)
  gives a real, testable bag-of-words vector with no model or network call,
  at the cost of being a *lexical* similarity measure: it cannot know "car"
  and "automobile" are related the way a real embedding model would, only
  that texts sharing more words are more similar.
- `OllamaEmbedder`: real embeddings from a small pre-trained model (e.g.
  `all-minilm`, ~45 MB, CPU-friendly) via Ollama's `/api/embed`. Needs
  `ollama pull all-minilm` and `ollama serve` running -- not verified against
  a live server while writing this (no Ollama process was running in this
  environment); validate before trusting it, the same caveat as
  `infergrid/queue/redpanda.py`.

The cache itself lives in the Phase 4 state store, one JSON list per tenant
(`semantic-cache:{tenant}`) rather than a key per entry: there is no way to
enumerate a key range in that store's `/kv` API (see DESIGN.md 3.5), and a
tenant's cache is small enough that "fetch the whole list, compare in
Python" is the honest, simple choice here -- a real deployment would use a
proper vector index. Capped at `max_entries`, oldest evicted first. Read-
modify-write against the store, not atomic: two gateway replicas racing to
add an entry for the same tenant at the same instant could lose one of them
-- acceptable for a cache (losing a cache write costs a future recompute,
never correctness), unlike `billing/consumer.py`'s usage totals.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from infergrid.common.hashring import ring_hash
from infergrid.store.client import StoreClient


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class HashEmbedder:
    def __init__(self, dims: int = 256):
        self.dims = dims

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for word in text.lower().split():
            vec[ring_hash(word) % self.dims] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec


class OllamaEmbedder:
    def __init__(self, http_client: httpx.AsyncClient, base_url: str = "http://127.0.0.1:11434",
                model: str = "all-minilm"):
        self.http_client = http_client
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def embed(self, text: str) -> list[float]:
        resp = await self.http_client.post(f"{self.base_url}/api/embed",
                                           json={"model": self.model, "input": text}, timeout=30.0)
        resp.raise_for_status()
        return resp.json()["embeddings"][0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


@dataclass
class CacheHit:
    content: str
    similarity: float


class SemanticCache:
    def __init__(self, store: StoreClient, embedder: Embedder, threshold: float = 0.92, max_entries: int = 200):
        self.store = store
        self.embedder = embedder
        self.threshold = threshold
        self.max_entries = max_entries

    async def lookup(self, tenant: str, text: str) -> CacheHit | None:
        entries = await self.store.get(f"semantic-cache:{tenant}") or []
        if not entries:
            return None
        query = await self.embedder.embed(text)
        best: CacheHit | None = None
        for entry in entries:
            sim = cosine_similarity(query, entry["embedding"])
            if sim >= self.threshold and (best is None or sim > best.similarity):
                best = CacheHit(entry["content"], sim)
        return best

    async def store_entry(self, tenant: str, text: str, content: str) -> None:
        key = f"semantic-cache:{tenant}"
        entries: list[dict[str, Any]] = await self.store.get(key) or []
        embedding = await self.embedder.embed(text)
        entries.append({"prompt": text, "embedding": embedding, "content": content, "stored_at": time.time()})
        if len(entries) > self.max_entries:
            entries = entries[-self.max_entries:]
        await self.store.put(key, entries)
