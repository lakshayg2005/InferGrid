from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from infergrid.common.schemas import GenerateRequest, GenerationResult


class BackendError(Exception):
    """The inference backend failed to produce a response."""


class Backend(ABC):
    """An inference engine a worker serves requests from."""

    name: str
    max_concurrency: int  # requests this backend can actually process at once; gossiped over SWIM
    max_queue: int  # admission limit: queue_depth() at or above this is refused, not queued

    @abstractmethod
    def generate(self, req: GenerateRequest, result: GenerationResult) -> AsyncIterator[str]:
        """Stream output text pieces, recording usage and finish reason in `result`."""

    @abstractmethod
    def queue_depth(self) -> int:
        """Requests currently active or waiting for a slot, checked before accepting a new one.

        This is admission control (load shedding): a worker at capacity refuses new work
        immediately with a clear error instead of queueing it indefinitely, so a client (or
        the gateway, on its behalf) can try another worker right away. Without it, requests
        pile up behind whatever is already running and everyone's latency degrades together.
        """

    @abstractmethod
    def stats(self) -> dict:
        """Load and cache statistics, reported to the gateway."""

    async def close(self) -> None:
        pass
