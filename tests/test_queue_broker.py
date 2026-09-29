"""Tests for infergrid/queue/broker/core.py: InferGrid's own from-scratch,
Kafka-style broker. No I/O here (Broker's methods are plain sync calls over
in-memory state) -- tests/test_queue_http.py covers the FastAPI + client layer
on top of it.
"""

import time

from infergrid.queue.broker.core import Broker


def test_publish_then_poll_returns_the_message():
    b = Broker(num_partitions=4)
    b.publish("t", key="a", value={"n": 1})
    msgs = b.poll("t", group="g", consumer_id="c1")
    assert len(msgs) == 1
    assert msgs[0].key == "a"
    assert msgs[0].value == {"n": 1}
    assert msgs[0].offset == 0


def test_same_key_always_lands_on_the_same_partition():
    b = Broker(num_partitions=8)
    partitions = {b.publish("t", key="tenant-x", value={}).partition for _ in range(10)}
    assert partitions == {next(iter(partitions))}


def test_a_message_is_not_redelivered_before_it_is_committed():
    b = Broker(num_partitions=1)
    b.publish("t", key="a", value={"n": 1})
    b.publish("t", key="b", value={"n": 2})  # same partition (only one exists)
    first = b.poll("t", group="g", consumer_id="c1")
    assert len(first) == 1  # the second message is behind the first, still in flight
    b.commit("t", "g", first[0])
    second = b.poll("t", group="g", consumer_id="c1")
    assert len(second) == 1
    assert second[0].key == "b"


def test_two_consumers_in_a_group_split_the_partitions():
    b = Broker(num_partitions=4)
    # Register both before publishing anything, so the partition assignment that
    # decides who gets what is already stable by the time there's real work --
    # a consumer joining mid-stream would only reshuffle *future* deliveries, not
    # reclaim a message already in flight to whoever held that partition before it.
    b.poll("t", group="g", consumer_id="c1", max_messages=0)
    b.poll("t", group="g", consumer_id="c2", max_messages=0)
    for i in range(20):
        b.publish("t", key=f"k{i}", value={"i": i})
    c1 = b.poll("t", group="g", consumer_id="c1", max_messages=100)
    c2 = b.poll("t", group="g", consumer_id="c2", max_messages=100)
    assert {m.partition for m in c1}.isdisjoint({m.partition for m in c2})
    assert len(c1) > 0 and len(c2) > 0
    assert len(c1) + len(c2) <= 20


def test_uncommitted_work_is_redelivered_after_its_consumer_goes_quiet():
    """The at-least-once guarantee: a consumer that polls a message and then
    never commits it (crashes) must not lose that message -- another consumer
    in the group must eventually get it."""
    b = Broker(num_partitions=1, session_timeout=0.05)
    b.publish("t", key="a", value={"n": 1})
    first = b.poll("t", group="g", consumer_id="c1")
    assert len(first) == 1
    time.sleep(0.1)  # c1 goes quiet without committing or polling again

    second = b.poll("t", group="g", consumer_id="c2")
    assert len(second) == 1
    assert second[0].key == "a"
    assert second[0].offset == first[0].offset


def test_committing_removes_the_consumer_from_holding_up_redelivery():
    b = Broker(num_partitions=1)
    b.publish("t", key="a", value={})
    [msg] = b.poll("t", group="g", consumer_id="c1")
    b.commit("t", "g", msg)
    # a second, unrelated consumer joining afterwards must not see stale work
    assert b.poll("t", group="g", consumer_id="c2") == []


def test_leave_frees_partitions_immediately_for_the_rest_of_the_group():
    b = Broker(num_partitions=2, session_timeout=100)
    b.publish("t", key="a", value={})  # ensure both consumers have joined the group once
    b.poll("t", group="g", consumer_id="c1", max_messages=0)
    b.poll("t", group="g", consumer_id="c2", max_messages=0)
    b.leave("t", "g", "c1")
    assert b.poll("t", group="g", consumer_id="c2", max_messages=100)[0].key == "a"


def test_different_groups_track_independent_offsets_on_the_same_topic():
    b = Broker(num_partitions=1)
    b.publish("t", key="a", value={})
    [msg] = b.poll("t", group="workers", consumer_id="w1")
    b.commit("t", "workers", msg)
    # a completely separate group (e.g. billing) must still see it from the start
    assert len(b.poll("t", group="billing", consumer_id="b1")) == 1
