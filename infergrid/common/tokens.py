"""Tokenisation and prefix block hashing.

Real LLMs use subword tokenisers; a word/punctuation split is close enough for
routing and simulation, and is identical on every node, which is what matters.

Prompts are split into fixed-size blocks, and each block's hash includes its
parent's hash. Two prompts therefore share block hash k exactly when their first
(k + 1) * BLOCK_SIZE tokens are identical. This is the scheme vLLM uses to
identify reusable KV-cache blocks, and it lets the router reason about cache
contents using nothing but hashes.
"""

import hashlib
import re
from collections.abc import Sequence

from infergrid.common.schemas import ChatMessage

BLOCK_SIZE = 16

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def render_prompt(messages: Sequence[ChatMessage]) -> str:
    """Flatten a conversation into one string.

    Rendering is prefix-stable: the rendering of the first k messages is a prefix
    of the rendering of all of them, so earlier chat turns become a cacheable prefix.
    """
    return "".join(f"<|{m.role}|>\n{m.content}\n" for m in messages)


def prompt_tokens(messages: Sequence[ChatMessage]) -> list[str]:
    return tokenize(render_prompt(messages))


def block_hashes(tokens: Sequence[str], block_size: int = BLOCK_SIZE) -> list[int]:
    """Chained hashes of every full block of tokens. A trailing partial block is not hashed."""
    hashes = []
    parent = b""
    for start in range(0, len(tokens) - block_size + 1, block_size):
        block = "\x1f".join(tokens[start : start + block_size]).encode()
        parent = hashlib.blake2b(parent + block, digest_size=8).digest()
        hashes.append(int.from_bytes(parent, "big"))
    return hashes


def routing_key(messages: Sequence[ChatMessage]) -> int:
    """Identify a conversation by its messages up to and including the first user message.

    Every later turn of the same conversation starts with exactly these messages, so
    the key stays the same for the whole conversation and all its turns reach the
    worker that already caches its history. Two alternatives fail:
      - the whole prompt changes every turn, so turns would scatter across workers;
      - the system prompt alone is shared by every conversation of an app, so a whole
        app's traffic would pile onto one worker.
    """
    first_user = next((i for i, m in enumerate(messages) if m.role == "user"), len(messages) - 1)
    return stable_seed(prompt_tokens(messages[: first_user + 1]))


def stable_seed(tokens: Sequence[str]) -> int:
    """A seed that is identical across processes (unlike Python's built-in hash())."""
    digest = hashlib.blake2b("\x1f".join(tokens).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")
