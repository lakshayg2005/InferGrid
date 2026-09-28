from infergrid.common.schemas import ChatMessage
from infergrid.common.tokens import block_hashes, prompt_tokens, render_prompt, tokenize


def words(n: int, prefix: str = "w") -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def test_shared_prefix_gives_shared_block_hashes():
    common = words(32)
    a = block_hashes(common + words(20, "a"))
    b = block_hashes(common + words(20, "b"))
    assert a[:2] == b[:2]
    assert a[2] != b[2]


def test_block_hash_depends_on_everything_before_it():
    # The second block is identical, but the first differs, so the second hash must differ too.
    a = block_hashes(words(16, "x") + words(16))
    b = block_hashes(words(16, "y") + words(16))
    assert a[1] != b[1]


def test_partial_block_is_not_hashed():
    assert len(block_hashes(words(15))) == 0
    assert len(block_hashes(words(47))) == 2


def test_rendering_is_prefix_stable():
    turns = [
        ChatMessage(role="system", content="Be brief."),
        ChatMessage(role="user", content="Hi there!"),
        ChatMessage(role="assistant", content="Hello."),
    ]
    assert render_prompt(turns).startswith(render_prompt(turns[:2]))
    assert prompt_tokens(turns)[: len(prompt_tokens(turns[:2]))] == prompt_tokens(turns[:2])


def test_tokenize_splits_words_and_punctuation():
    assert tokenize("Hello, world!") == ["Hello", ",", "world", "!"]
