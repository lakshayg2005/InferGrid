import asyncio
import json

import httpx

from infergrid.common.schemas import ChatMessage, GenerateRequest, GenerationResult
from infergrid.worker.app import create_app
from infergrid.worker.backends import SimBackend, SimConfig
from infergrid.worker.prefix_cache import PrefixCache

FAST = SimConfig(prefill_ms_per_token=0, decode_ms_per_token=0)
SYSTEM = ChatMessage(role="system", content=" ".join(f"rule{i}" for i in range(64)))


def request(user: str, max_tokens: int | None = None) -> GenerateRequest:
    return GenerateRequest(
        request_id="r", messages=[SYSTEM, ChatMessage(role="user", content=user)], max_tokens=max_tokens
    )


async def run(backend: SimBackend, req: GenerateRequest) -> tuple[str, GenerationResult]:
    result = GenerationResult()
    text = "".join([piece async for piece in backend.generate(req, result)])
    return text, result


def test_prefix_cache_counts_leading_blocks_only():
    cache = PrefixCache(capacity_blocks=100, block_size=4)
    assert cache.lookup_and_insert(list("abcdefgh")) == 0
    assert cache.lookup_and_insert(list("abcdefghij")) == 8
    assert cache.lookup_and_insert(list("abcdXXXX")) == 4


def test_prefix_cache_evicts_least_recently_used():
    cache = PrefixCache(capacity_blocks=2, block_size=4)
    cache.lookup_and_insert(list("aaaa"))
    cache.lookup_and_insert(list("bbbb"))
    cache.lookup_and_insert(list("cccc"))  # evicts "aaaa"
    assert cache.lookup_and_insert(list("aaaa")) == 0
    assert cache.lookup_and_insert(list("cccc")) == 4


async def test_sim_reuses_shared_prefix():
    backend = SimBackend(FAST)
    _, first = await run(backend, request("question one"))
    _, second = await run(backend, request("a different question"))
    assert first.usage.cached_tokens == 0
    assert second.usage.cached_tokens >= 64  # the long system prompt was cached


async def test_sim_caches_its_own_reply_for_the_next_turn():
    backend = SimBackend(FAST)
    turn1 = request("first question")
    reply, first = await run(backend, turn1)
    turn2 = GenerateRequest(request_id="r2", messages=[
        *turn1.messages,
        ChatMessage(role="assistant", content=reply),
        ChatMessage(role="user", content="second question"),
    ])
    _, second = await run(backend, turn2)
    # Everything up to the new user message was seen before, apart from a partial last block.
    seen_before = first.usage.prompt_tokens + first.usage.completion_tokens
    assert second.usage.cached_tokens > seen_before - 16


async def test_sim_is_deterministic_and_respects_max_tokens():
    text_a, _ = await run(SimBackend(FAST), request("same prompt"))
    text_b, _ = await run(SimBackend(FAST), request("same prompt"))
    assert text_a == text_b

    _, result = await run(SimBackend(FAST), request("same prompt", max_tokens=3))
    assert result.usage.completion_tokens == 3
    assert result.finish_reason == "length"


async def test_sim_limits_concurrency():
    backend = SimBackend(SimConfig(prefill_ms_per_token=0, decode_ms_per_token=5, max_concurrency=2))
    tasks = [asyncio.create_task(run(backend, request(f"q{i}", max_tokens=5))) for i in range(5)]
    await asyncio.sleep(0.01)
    assert backend.stats()["active"] == 2
    assert backend.stats()["queued"] == 3
    await asyncio.gather(*tasks)
    assert backend.stats()["active"] == 0


async def test_worker_streams_indexed_tokens_then_done():
    app = create_app(SimBackend(FAST), "w1")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://w1") as client:
        resp = await client.post("/generate", json=request("hi", max_tokens=4).model_dump())
    events = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
    assert [e["index"] for e in events[:-1]] == [0, 1, 2, 3]
    assert events[-1]["type"] == "done"
    assert events[-1]["usage"]["completion_tokens"] == 4


async def test_sim_resumes_an_answer_from_a_token_index():
    full, _ = await run(SimBackend(FAST), request("resume me", max_tokens=10))
    pieces = [full.split(" ")[0]] + [" " + w for w in full.split(" ")[1:]]
    resumed_req = request("resume me", max_tokens=10).model_copy(
        update={"resume_text": "".join(pieces[:4]), "resume_tokens": 4})
    rest, result = await run(SimBackend(FAST), resumed_req)
    assert "".join(pieces[:4]) + rest == full
    assert result.usage.completion_tokens == 10  # counts the whole answer
    assert result.usage.prompt_tokens > 0


async def test_worker_numbers_resumed_tokens_from_the_resume_point():
    app = create_app(SimBackend(FAST), "w1")
    req = request("hi", max_tokens=6).model_copy(update={"resume_text": "a b", "resume_tokens": 2})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://w1") as client:
        resp = await client.post("/generate", json=req.model_dump())
    events = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
    assert [e["index"] for e in events[:-1]] == [2, 3, 4, 5]


async def test_membership_endpoint_empty_when_no_membership_configured():
    app = create_app(SimBackend(FAST), "w1")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://w1") as client:
        resp = await client.get("/membership")
    assert resp.json() == {}


async def test_worker_reports_its_own_membership_snapshot():
    from infergrid.membership import SwimNode
    node = SwimNode("127.0.0.1:19501", metadata={"http_url": "http://w1"},
                    protocol_period=0.05, ping_timeout=0.05, suspicion_timeout=0.2)
    app = create_app(SimBackend(FAST), "w1", membership=node)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://w1") as client:
        resp = await client.get("/membership")
    snapshot = resp.json()
    assert snapshot["127.0.0.1:19501"]["state"] == "alive"
    assert snapshot["127.0.0.1:19501"]["metadata"] == {"http_url": "http://w1"}
