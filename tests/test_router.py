import math

import pytest

from infergrid.common.schemas import ChatMessage, GenerateRequest
from infergrid.gateway.router import ConsistentHashRouter, LeastLoadedRouter, RoundRobinRouter, make_router

WORKERS = ["w1", "w2", "w3", "w4"]
IDLE = {w: 0 for w in WORKERS}


def chat(first_question: str, extra_turns: int = 0) -> GenerateRequest:
    messages = [ChatMessage(role="system", content="You are a shopping assistant."),
                ChatMessage(role="user", content=first_question)]
    for i in range(extra_turns):
        messages += [ChatMessage(role="assistant", content=f"answer {i}"),
                     ChatMessage(role="user", content=f"follow-up {i}")]
    return GenerateRequest(request_id="r", messages=messages)


def test_round_robin_rotates_through_all_workers():
    router = RoundRobinRouter()
    req = chat("x")
    assert router.candidates(req, ["a", "b", "c"], {}) == ["a", "b", "c"]
    assert router.candidates(req, ["a", "b", "c"], {}) == ["b", "c", "a"]
    assert router.candidates(req, ["a", "b", "c"], {}) == ["c", "a", "b"]
    assert router.candidates(req, [], {}) == []


def test_least_loaded_prefers_idle_worker_and_rotates_ties():
    router = LeastLoadedRouter()
    assert router.candidates(chat("x"), WORKERS, {"w1": 3, "w2": 1, "w3": 0, "w4": 2})[0] == "w3"
    firsts = {router.candidates(chat("x"), WORKERS, IDLE)[0] for _ in range(4)}
    assert firsts == set(WORKERS)


def test_consistent_hash_keeps_a_conversation_on_one_worker():
    router = ConsistentHashRouter()
    home = router.candidates(chat("do you sell shoes?"), WORKERS, IDLE)[0]
    for turns in range(1, 6):
        assert router.candidates(chat("do you sell shoes?", turns), WORKERS, IDLE)[0] == home


def test_consistent_hash_spreads_different_conversations():
    router = ConsistentHashRouter()
    homes = {router.candidates(chat(f"question {i}"), WORKERS, IDLE)[0] for i in range(100)}
    assert homes == set(WORKERS)


def test_bounded_load_skips_a_full_worker():
    router = ConsistentHashRouter(epsilon=0.25)
    req = chat("popular question")
    home = router.candidates(req, WORKERS, IDLE)[0]
    # 8 requests in flight, all on `home`. Average with this one is 9/4, so the cap is ceil(1.25 * 2.25) = 3.
    load = {**IDLE, home: 8}
    choice = router.candidates(req, WORKERS, load)
    assert choice[0] != home
    assert choice[-1] == home  # the full worker is the last resort
    assert sorted(choice) == sorted(WORKERS)


def test_bounded_load_allows_imbalance_up_to_the_cap():
    router = ConsistentHashRouter(epsilon=0.25)
    req = chat("popular question")
    home = router.candidates(req, WORKERS, IDLE)[0]
    # 8 other requests, 2 on home: cap is ceil(1.25 * 9 / 4) = 3, so home still has room.
    load = {w: 2 for w in WORKERS}
    assert router.candidates(req, WORKERS, load)[0] == home


def test_unbounded_consistent_hash_ignores_load():
    router = ConsistentHashRouter(epsilon=math.inf)
    req = chat("popular question")
    home = router.candidates(req, WORKERS, IDLE)[0]
    assert router.candidates(req, WORKERS, {**IDLE, home: 100})[0] == home


def test_consistent_hash_follows_worker_set_changes():
    router = ConsistentHashRouter()
    reqs = [chat(f"question {i}") for i in range(200)]
    before = {i: router.candidates(r, WORKERS, IDLE)[0] for i, r in enumerate(reqs)}
    fewer = [w for w in WORKERS if w != "w2"]
    after = {i: router.candidates(r, fewer, {w: 0 for w in fewer})[0] for i, r in enumerate(reqs)}
    assert "w2" not in after.values()
    assert all(after[i] == before[i] for i in before if before[i] != "w2")


def test_make_router():
    assert isinstance(make_router("round_robin"), RoundRobinRouter)
    assert make_router("consistent_hash", 0.5).epsilon == 0.5
    with pytest.raises(ValueError):
        make_router("random")
