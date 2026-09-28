"""A real pre-trained model served by a local Ollama instance (CPU is fine)."""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from infergrid.common.schemas import GenerateRequest, GenerationResult
from infergrid.worker.backends.base import Backend, BackendError


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", max_concurrency: int = 2):
        self.model = model
        self.max_concurrency = max_concurrency
        # Loading a model into memory can take a while on CPU, hence the long read timeout.
        self._client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(10.0, read=300.0))
        self._slots = asyncio.Semaphore(max_concurrency)
        self._active = 0
        self._queued = 0
        self._requests = 0

    async def generate(self, req: GenerateRequest, result: GenerationResult) -> AsyncIterator[str]:
        payload: dict = {
            "model": self.model,
            "messages": [m.model_dump() for m in req.messages],
            "stream": True,
        }
        if req.max_tokens:
            payload["options"] = {"num_predict": req.max_tokens}

        self._queued += 1
        try:
            await self._slots.acquire()
        finally:
            self._queued -= 1

        self._active += 1
        self._requests += 1
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")
                    raise BackendError(f"ollama returned {resp.status_code}: {body[:200]}")
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise BackendError(chunk["error"])
                    text = chunk.get("message", {}).get("content", "")
                    if text:
                        result.usage.completion_tokens += 1
                        yield text
                    if chunk.get("done"):
                        # Ollama does not report prefix-cache hits, so cached_tokens stays 0.
                        result.usage.prompt_tokens = chunk.get("prompt_eval_count", 0)
                        result.usage.completion_tokens = chunk.get("eval_count", result.usage.completion_tokens)
                        result.finish_reason = "length" if chunk.get("done_reason") == "length" else "stop"
        except httpx.TransportError as exc:
            raise BackendError(f"cannot reach ollama: {exc!r}") from exc
        finally:
            self._active -= 1
            self._slots.release()

    def stats(self) -> dict:
        return {
            "backend": self.name,
            "model": self.model,
            "active": self._active,
            "queued": self._queued,
            "max_concurrency": self.max_concurrency,
            "requests_total": self._requests,
        }

    async def close(self) -> None:
        await self._client.aclose()
