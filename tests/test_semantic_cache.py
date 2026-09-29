"""Tests for infergrid/semantic_cache.py. A tiny in-memory `FakeStore` stands in
for `store.client.StoreClient` (only `get`/`put` are used, so anything with
that shape works) -- this project's own store already has its own tests
(tests/test_store.py, tests/test_store_http.py); this file is about the
caching logic layered on top of it.
"""

import pytest

from infergrid.semantic_cache import HashEmbedder, SemanticCache, cosine_similarity


class FakeStore:
    def __init__(self):
        self.data = {}

    async def get(self, key):
        return self.data.get(key)

    async def put(self, key, value):
        self.data[key] = value


# -- cosine_similarity -----------------------------------------------------


def test_identical_vectors_have_similarity_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_a_zero_vector_has_similarity_zero_not_a_division_error():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# -- HashEmbedder ------------------------------------------------------------


async def test_hash_embedder_is_deterministic_across_instances():
    a, b = HashEmbedder(), HashEmbedder()
    assert await a.embed("how do I reset my password") == await b.embed("how do I reset my password")


async def test_hash_embedder_is_insensitive_to_word_order():
    e = HashEmbedder()
    v1 = await e.embed("reset my password please")
    v2 = await e.embed("please reset my password")
    assert cosine_similarity(v1, v2) == pytest.approx(1.0)


async def test_hash_embedder_rates_overlapping_text_more_similar_than_unrelated_text():
    e = HashEmbedder()
    base = await e.embed("how do I reset my account password")
    similar = await e.embed("how do I reset my password")
    unrelated = await e.embed("what is the weather forecast tomorrow")
    assert cosine_similarity(base, similar) > cosine_similarity(base, unrelated)


# -- SemanticCache -------------------------------------------------------------


async def test_lookup_on_an_empty_cache_is_a_miss():
    cache = SemanticCache(FakeStore(), HashEmbedder())
    assert await cache.lookup("acme", "hello") is None


async def test_an_exact_repeat_is_a_cache_hit():
    cache = SemanticCache(FakeStore(), HashEmbedder())
    await cache.store_entry("acme", "how do I reset my password?", "click forgot password")
    hit = await cache.lookup("acme", "how do I reset my password?")
    assert hit is not None
    assert hit.content == "click forgot password"
    assert hit.similarity == pytest.approx(1.0)


async def test_an_unrelated_prompt_is_a_miss():
    cache = SemanticCache(FakeStore(), HashEmbedder(), threshold=0.92)
    await cache.store_entry("acme", "how do I reset my password?", "click forgot password")
    assert await cache.lookup("acme", "what is the weather like today?") is None


async def test_cache_entries_are_scoped_per_tenant():
    cache = SemanticCache(FakeStore(), HashEmbedder())
    await cache.store_entry("acme", "how do I reset my password?", "acme's answer")
    assert await cache.lookup("globex", "how do I reset my password?") is None


async def test_a_lower_threshold_accepts_a_looser_paraphrase():
    cache_strict = SemanticCache(FakeStore(), HashEmbedder(), threshold=0.99)
    cache_loose = SemanticCache(FakeStore(), HashEmbedder(), threshold=0.5)
    for cache in (cache_strict, cache_loose):
        await cache.store_entry("acme", "how do I reset my account password", "click forgot password")
    assert await cache_strict.lookup("acme", "how do I reset my password") is None
    assert await cache_loose.lookup("acme", "how do I reset my password") is not None


async def test_max_entries_evicts_the_oldest_first():
    store = FakeStore()
    cache = SemanticCache(store, HashEmbedder(), max_entries=2)
    await cache.store_entry("acme", "question one", "answer one")
    await cache.store_entry("acme", "question two", "answer two")
    await cache.store_entry("acme", "question three", "answer three")
    entries = store.data["semantic-cache:acme"]
    assert len(entries) == 2
    assert [e["prompt"] for e in entries] == ["question two", "question three"]
    assert await cache.lookup("acme", "question one") is None
