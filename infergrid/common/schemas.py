"""Messages exchanged between clients, the gateway and workers."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Public API request, a subset of OpenAI's /v1/chat/completions.

    Unknown fields (temperature, top_p, ...) are ignored, so OpenAI SDKs work unchanged.
    """

    model: str = "infergrid"
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    stream: bool = False


class GenerateRequest(BaseModel):
    """Internal request from the gateway to a worker."""

    request_id: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    cached_tokens: int = 0  # prompt tokens served from the worker's prefix cache
    completion_tokens: int = 0


@dataclass
class GenerationResult:
    """Filled in by a backend while it streams, read by the worker when the stream ends."""

    usage: Usage = field(default_factory=Usage)
    finish_reason: Literal["stop", "length"] = "stop"
