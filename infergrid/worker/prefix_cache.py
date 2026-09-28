"""A block-level LRU cache that models an LLM worker's KV cache."""

from collections import OrderedDict
from collections.abc import Sequence

from infergrid.common.tokens import BLOCK_SIZE, block_hashes


class PrefixCache:
    def __init__(self, capacity_blocks: int, block_size: int = BLOCK_SIZE):
        self.capacity_blocks = capacity_blocks
        self.block_size = block_size
        self._blocks: OrderedDict[int, None] = OrderedDict()
        self.hit_tokens = 0
        self.prompt_tokens = 0

    def lookup_and_insert(self, tokens: Sequence[str]) -> int:
        """Return how many leading tokens were already cached, then cache the whole prompt."""
        hashes = block_hashes(tokens, self.block_size)

        matched = 0
        for h in hashes:
            if h not in self._blocks:
                break
            matched += 1

        self._insert(hashes)
        cached = matched * self.block_size
        self.hit_tokens += cached
        self.prompt_tokens += len(tokens)
        return cached

    def insert(self, tokens: Sequence[str]) -> None:
        """Cache a sequence without counting it as a lookup, e.g. a prompt plus its generated reply."""
        self._insert(block_hashes(tokens, self.block_size))

    def _insert(self, hashes: list[int]) -> None:
        # Touch blocks from last to first so a sequence's first block is the most
        # recently used. Eviction then removes the tail of a prefix before its head;
        # evicting a head first would make every block after it unreachable.
        for h in reversed(hashes):
            self._blocks[h] = None
            self._blocks.move_to_end(h)
        while len(self._blocks) > self.capacity_blocks:
            self._blocks.popitem(last=False)

    def stats(self) -> dict:
        return {
            "blocks": len(self._blocks),
            "capacity_blocks": self.capacity_blocks,
            "hit_tokens": self.hit_tokens,
            "prompt_tokens": self.prompt_tokens,
            "hit_rate": self.hit_tokens / self.prompt_tokens if self.prompt_tokens else 0.0,
        }
