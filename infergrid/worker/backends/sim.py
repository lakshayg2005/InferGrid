"""A simulated LLM, so benchmarks can run many workers on one laptop.

The latency model follows how real inference engines behave:
  - prefill cost is paid only for prompt tokens missing from the prefix cache,
  - generated tokens are cached too, so the next turn of a chat can reuse the
    whole conversation so far, including the model's own previous reply,
  - decode cost is paid per generated token,
  - at most `max_concurrency` requests run at once; the rest queue.

Real engines batch decode steps across requests; here every running request decodes
independently, which is a simplification that does not change routing decisions.

Output is deterministic for a given prompt, which keeps tests and benchmarks repeatable.
"""

import asyncio
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass

from infergrid.common.schemas import ChatMessage, GenerateRequest, GenerationResult
from infergrid.common.tokens import BLOCK_SIZE, prompt_tokens, stable_seed
from infergrid.worker.backends.base import Backend
from infergrid.worker.prefix_cache import PrefixCache

_VOCAB = (
    "the system routes each request to a worker that already holds its prefix "
    "so the cluster avoids recomputing shared context and keeps latency low "
    "replicas agree on state while shards spread the load across many nodes"
).split()


@dataclass
class SimConfig:
    prefill_ms_per_token: float = 0.4
    decode_ms_per_token: float = 25.0
    max_concurrency: int = 4
    cache_capacity_blocks: int = 4096
    block_size: int = BLOCK_SIZE
    min_output_tokens: int = 24
    max_output_tokens: int = 160


class SimBackend(Backend):
    name = "sim"

    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()
        self.cache = PrefixCache(self.config.cache_capacity_blocks, self.config.block_size)
        self._slots = asyncio.Semaphore(self.config.max_concurrency)
        self._active = 0
        self._queued = 0
        self._requests = 0

    async def generate(self, req: GenerateRequest, result: GenerationResult) -> AsyncIterator[str]:
        cfg = self.config
        tokens = prompt_tokens(req.messages)
        rng = random.Random(stable_seed(tokens))
        natural_length = rng.randint(cfg.min_output_tokens, cfg.max_output_tokens)
        length = min(natural_length, req.max_tokens) if req.max_tokens else natural_length

        self._queued += 1
        try:
            await self._slots.acquire()
        finally:
            self._queued -= 1

        self._active += 1
        self._requests += 1
        try:
            cached = self.cache.lookup_and_insert(tokens)
            result.usage.prompt_tokens = len(tokens)
            result.usage.cached_tokens = cached
            await asyncio.sleep((len(tokens) - cached) * cfg.prefill_ms_per_token / 1000)

            pieces = []
            for i in range(length):
                await asyncio.sleep(cfg.decode_ms_per_token / 1000)
                pieces.append(rng.choice(_VOCAB) if i == 0 else " " + rng.choice(_VOCAB))
                result.usage.completion_tokens += 1
                yield pieces[-1]
            result.finish_reason = "length" if length < natural_length else "stop"

            reply = ChatMessage(role="assistant", content="".join(pieces))
            self.cache.insert(prompt_tokens([*req.messages, reply]))
        finally:
            self._active -= 1
            self._slots.release()

    def stats(self) -> dict:
        return {
            "backend": self.name,
            "active": self._active,
            "queued": self._queued,
            "max_concurrency": self.config.max_concurrency,
            "requests_total": self._requests,
            "cache": self.cache.stats(),
        }
