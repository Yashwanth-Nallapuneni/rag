"""Token counting.

Chunk sizes are specified in tokens, not characters, because that is the unit
the embedding model and the LLM context window actually operate in. A single
shared encoder here keeps chunking, context packing and eval reporting all
counting the same way -- if they disagree, "650-token chunks" silently stops
meaning anything.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol


class Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


class _WhitespaceEncoder:
    """Fallback when tiktoken is unavailable or offline.

    Approximates BPE at roughly 4 chars/token so chunk sizes stay in the right
    ballpark. Used only as a last resort; it is not byte-accurate.
    """

    def encode(self, text: str) -> list[int]:
        return list(range(max(1, len(text) // 4))) if text else []

    def decode(self, tokens: list[int]) -> str:  # pragma: no cover
        raise NotImplementedError("fallback encoder cannot decode")


class _TiktokenEncoder:
    """Wraps a tiktoken encoding to treat special tokens as ordinary text.

    Real documents contain these strings: a paper about fill-in-the-middle
    code models quotes "<|fim_middle|>" verbatim. tiktoken raises on such
    tokens by default, which would abort an entire corpus ingest over one
    quoted string, so the check is disabled here rather than at each call
    site.
    """

    def __init__(self, encoding):
        self._enc = encoding

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)


@lru_cache(maxsize=4)
def get_encoder(name: str = "cl100k_base") -> Encoder:
    try:
        import tiktoken

        return _TiktokenEncoder(tiktoken.get_encoding(name))
    except Exception:  # noqa: BLE001 - tiktoken may need network on first use
        return _WhitespaceEncoder()


def count_tokens(text: str, encoder_name: str = "cl100k_base") -> int:
    if not text:
        return 0
    return len(get_encoder(encoder_name).encode(text))


def truncate_to_tokens(
    text: str, max_tokens: int, encoder_name: str = "cl100k_base"
) -> str:
    """Cut text to a token budget, decoding back to a string when possible."""
    enc = get_encoder(encoder_name)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    try:
        return enc.decode(tokens[:max_tokens])
    except NotImplementedError:
        return text[: max_tokens * 4]
