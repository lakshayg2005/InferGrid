"""A realistic chat workload for benchmarks and chaos tests.

  - a few "apps", each with its own long system prompt shared by all its users,
  - conversations of several turns, where each turn resends the whole history,
  - new conversations arrive as a Poisson process, users pause between turns.

Conversations start on a fixed schedule whether or not earlier requests have
finished (an "open-loop" load generator). A closed loop, where each client waits
for its previous response, would quietly send less traffic whenever the system
slowed down and so hide exactly the queueing delays we want to measure.
"""

import asyncio
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

_WORDS = (
    "order return refund size fit colour cotton denim sneaker kurta saree jacket discount "
    "delivery exchange wishlist cart payment coupon brand style trend summer winter party "
    "office casual formal budget premium review rating stock warehouse pincode tracking"
).split()


@dataclass
class Conversation:
    start: float  # seconds after the run starts
    system: str
    user_messages: list[str]
    think_times: list[float]  # pause after each reply before the next message


@dataclass
class Record:
    conversation: int
    turn: int
    ok: bool = False
    sent_at: float = 0.0  # seconds after the run started
    worker: str = ""  # the worker that started the answer
    ttft: float = 0.0  # seconds from sending the request to the first content token
    latency: float = 0.0  # seconds until the last token
    prompt_tokens: int = 0
    cached_tokens: int = 0
    finish_reason: str = ""
    error: str = ""


# Called after each successful turn with the record, the messages sent and the reply received.
ReplyCheck = Callable[[Record, list[dict], str], None]


def build_workload(seed: int, duration: float, rate: float, apps: int) -> list[Conversation]:
    rng = random.Random(seed)
    systems = [
        f"You are the assistant for app {a}. " + " ".join(rng.choices(_WORDS, k=rng.randint(250, 700)))
        for a in range(apps)
    ]
    conversations, t = [], 0.0
    while True:
        t += rng.expovariate(rate)
        if t > duration:
            return conversations
        turns = rng.randint(2, 8)
        conversations.append(Conversation(
            start=t,
            system=rng.choice(systems),
            user_messages=[" ".join(rng.choices(_WORDS, k=rng.randint(8, 50))) for _ in range(turns)],
            think_times=[rng.uniform(0.5, 2.0) for _ in range(turns)],
        ))


async def one_request(client: httpx.AsyncClient, url: str, messages: list[dict], max_tokens: int,
                      rec: Record) -> str:
    sent = time.perf_counter()
    parts = []
    body = {"messages": messages, "stream": True, "max_tokens": max_tokens}
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                rec.error = f"HTTP {resp.status_code}"
                return ""
            rec.worker = resp.headers.get("x-infergrid-worker", "")
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if "error" in chunk:
                    rec.error = chunk["error"]["message"]
                    return ""
                choice = chunk["choices"][0]
                if text := choice["delta"].get("content"):
                    if not parts:
                        rec.ttft = time.perf_counter() - sent
                    parts.append(text)
                if choice["finish_reason"]:
                    rec.finish_reason = choice["finish_reason"]
                if usage := chunk.get("usage"):
                    rec.prompt_tokens = usage["prompt_tokens"]
                    rec.cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
    except httpx.HTTPError as exc:
        rec.error = repr(exc)
        return ""
    if not rec.finish_reason:
        rec.error = "stream ended without finishing"
        return ""
    rec.latency = time.perf_counter() - sent
    rec.ok = True
    return "".join(parts)


async def run_conversation(client: httpx.AsyncClient, url: str, cid: int, conv: Conversation, started: float,
                           max_tokens: int, records: list[Record], check: ReplyCheck | None = None) -> None:
    await asyncio.sleep(max(0.0, started + conv.start - time.perf_counter()))
    history = [{"role": "system", "content": conv.system}]
    for turn, (message, think) in enumerate(zip(conv.user_messages, conv.think_times)):
        history.append({"role": "user", "content": message})
        rec = Record(conversation=cid, turn=turn, sent_at=time.perf_counter() - started)
        reply = await one_request(client, url, history, max_tokens, rec)
        records.append(rec)
        if not rec.ok:
            return
        if check:
            check(rec, list(history), reply)
        history.append({"role": "assistant", "content": reply})
        await asyncio.sleep(think)


async def run_workload(url: str, workload: list[Conversation], max_tokens: int,
                       check: ReplyCheck | None = None) -> tuple[list[Record], float]:
    records: list[Record] = []
    limits = httpx.Limits(max_connections=1000, max_keepalive_connections=200)
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=120.0), limits=limits) as client:
        started = time.perf_counter()
        await asyncio.gather(*(run_conversation(client, url, i, conv, started, max_tokens, records, check)
                               for i, conv in enumerate(workload)))
        return records, time.perf_counter() - started


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]
