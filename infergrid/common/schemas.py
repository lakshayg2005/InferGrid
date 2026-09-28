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
    """Internal request from the gateway to a worker.

    When a worker dies mid-answer, the gateway sends the same request to another
    worker with the part of the answer the client already has. The new worker
    continues from there and numbers its tokens from `resume_tokens` onwards.
    """

    request_id: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = None  # for the whole answer, including any resumed part
    resume_text: str = ""
    resume_tokens: int = Field(default=0, ge=0)


class Usage(BaseModel):
    prompt_tokens: int = 0
    cached_tokens: int = 0  # prompt tokens served from the worker's prefix cache
    completion_tokens: int = 0  # the whole answer, including any part resumed from another worker


@dataclass
class GenerationResult:
    """Filled in by a backend while it streams, read by the worker when the stream ends."""

    usage: Usage = field(default_factory=Usage)
    finish_reason: Literal["stop", "length"] = "stop"
