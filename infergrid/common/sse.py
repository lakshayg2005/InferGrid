"""Server-Sent Events encoding and decoding."""

import json
from collections.abc import AsyncIterator
from typing import Any


def encode(data: dict[str, Any] | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    return f"data: {payload}\n\n"


async def iter_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield the payload of every `data:` line in an SSE stream."""
    async for line in lines:
        if line.startswith("data:"):
            yield line[5:].strip()
