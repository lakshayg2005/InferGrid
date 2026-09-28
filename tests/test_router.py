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


# -- capacity-aware bound (set_capacities) ---------------------------------------

def test_capacity_bound_ignores_worker_count_unlike_the_average_bound():
    """The bug this fixes: losing a worker from the candidate set should not, by
    itself, raise what the survivors are allowed to carry."""
    router = ConsistentHashRouter(epsilon=0.25)
    router.set_capacities({w: 4 for w in WORKERS})  # every worker can really do 4 at once
    req = chat("popular question")
    home = router.candidates(req, WORKERS, IDLE)[0]

    # cap is ceil(1.25 * 4) = 5, independent of how many workers are in the candidate set.
    load = {**IDLE, home: 5}
    assert router.candidates(req, WORKERS, load)[0] != home
    fewer = [w for w in WORKERS if w != next(w for w in WORKERS if w != home)]
    assert router.candidates(req, fewer, load)[0] != home  # still excluded with fewer peers


def test_capacity_bound_gives_a_bigger_worker_more_headroom():
    router = ConsistentHashRouter(epsilon=0.25)
    router.set_capacities({"w1": 8, "w2": 2, "w3": 4, "w4": 4})
    req = chat("popular question")
    # Force w1 to be tried first regardless of its ring position, by giving it 0 load
    # and everyone else a load already at or above their own cap.
    load = {"w1": 9, "w2": 2, "w3": 4, "w4": 4}  # ceil(1.25*8)=10, ceil(1.25*2)=3, ceil(1.25*4)=5
    order = router.candidates(req, WORKERS, load)
    # w1 (load 9 < cap 10) has room; w2 (load 2 < cap 3) has room too; w3, w4 are at cap.
    assert order[0] in {"w1", "w2"}
    assert order[-1] in {"w3", "w4"} or order[-2] in {"w3", "w4"}


def test_capacity_bound_falls_back_to_average_when_unset():
    router = ConsistentHashRouter(epsilon=0.25)
    req = chat("popular question")
    home = router.candidates(req, WORKERS, IDLE)[0]
    load = {**IDLE, home: 8}  # matches test_bounded_load_skips_a_full_worker's average-based case
    assert router.candidates(req, WORKERS, load)[0] != home


def test_set_capacities_empty_dict_reverts_to_average_bound():
    router = ConsistentHashRouter(epsilon=0.25)
    router.set_capacities({w: 100 for w in WORKERS})  # huge capacity: nothing would ever be excluded
    router.set_capacities({})  # revert
    req = chat("popular question")
    home = router.candidates(req, WORKERS, IDLE)[0]
    load = {**IDLE, home: 8}
    assert router.candidates(req, WORKERS, load)[0] != home  # average-based bound is back in effect
