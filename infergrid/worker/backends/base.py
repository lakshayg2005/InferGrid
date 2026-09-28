from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from infergrid.common.schemas import GenerateRequest, GenerationResult


class BackendError(Exception):
    """The inference backend failed to produce a response."""


class Backend(ABC):
    """An inference engine a worker serves requests from."""

    name: str

    @abstractmethod
    def generate(self, req: GenerateRequest, result: GenerationResult) -> AsyncIterator[str]:
        """Stream output text pieces, recording usage and finish reason in `result`."""

    @abstractmethod
    def stats(self) -> dict:
        """Load and cache statistics, reported to the gateway."""

    async def close(self) -> None:
        pass
