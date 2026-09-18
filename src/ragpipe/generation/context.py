"""Rendering retrieved chunks into a prompt context block.

The format is a contract, not a cosmetic choice: the marker assigned here is
what the model cites, what `citations.py` resolves back to a chunk, and what
the offline mock LLM parses. Changing the layout without updating all three
breaks citation resolution silently, so the regex that reads it lives next to
the writer that produces it.

Markers are `[S1]`, not `[1]`, and the `S` is load-bearing. Academic prose is
dense with its own inline reference markers -- "generative retrieval in
industrial search [4, 15, 20, 26]" -- and a plain `[n]` scheme cannot tell
those apart from a citation to passage n. Quoting a passage verbatim then
produces an answer that appears to cite sources 4, 15, 20 and 26, which may
not even exist in the context. `[S...]` cannot collide with a paper's own
bibliography numbering.

Layout, one blank line between blocks:

    [S1] (Paper Title | 3.2 Attention | p. 4)
    ...chunk text...

    [S2] (Other Paper | p. 1)
    ...chunk text...
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas import Citation, RetrievedChunk
from ..tokenization import count_tokens, truncate_to_tokens

# Must stay in sync with the block layout written by `render_context`.
MARKER_PREFIX = "S"
BLOCK_HEADER_RE = re.compile(
    rf"^\s*\[{MARKER_PREFIX}(\d+)\]\s*(?:\(([^)]*)\))?\s*$", re.MULTILINE
)


def format_marker(n: int) -> str:
    return f"[{MARKER_PREFIX}{n}]"


@dataclass
class RenderedContext:
    text: str
    used: list[RetrievedChunk]
    dropped: list[RetrievedChunk]
    tokens: int

    @property
    def marker_to_chunk(self) -> dict[int, RetrievedChunk]:
        """Markers are 1-based and assigned in the order passed in."""
        return {i + 1: rc for i, rc in enumerate(self.used)}


def render_context(
    retrieved: list[RetrievedChunk],
    max_tokens: int,
    *,
    include_locators: bool = True,
) -> RenderedContext:
    """Pack as many retrieved chunks as fit into the context budget.

    Chunks are added in rank order and a chunk that does not fit is dropped
    rather than truncated, so the model never sees half a passage and cites it
    as if it were whole. The one exception is a single chunk larger than the
    entire budget, which is truncated because dropping it would leave no
    context at all.
    """
    blocks: list[str] = []
    used: list[RetrievedChunk] = []
    dropped: list[RetrievedChunk] = []
    total = 0

    for rc in retrieved:
        marker = len(used) + 1
        tag = format_marker(marker)
        header = f"{tag} ({rc.chunk.locator()})" if include_locators else tag
        body = rc.chunk.text.strip()
        block = f"{header}\n{body}"
        cost = count_tokens(block) + 1

        if total + cost > max_tokens:
            if not used:
                # Nothing packed yet: truncate rather than return no context.
                budget = max(50, max_tokens - count_tokens(header) - 8)
                block = f"{header}\n{truncate_to_tokens(body, budget)}"
                blocks.append(block)
                used.append(rc)
                total = count_tokens(block)
            else:
                dropped.append(rc)
            continue

        blocks.append(block)
        used.append(rc)
        total += cost

    return RenderedContext(
        text="\n\n".join(blocks), used=used, dropped=dropped, tokens=total
    )


def citation_for(marker: int, rc: RetrievedChunk, quote: str | None = None) -> Citation:
    chunk = rc.chunk
    return Citation(
        marker=marker,
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        doc_title=chunk.doc_title,
        locator=chunk.locator(),
        page_start=chunk.page_start,
        section_path=list(chunk.section_path),
        source_uri=chunk.source_uri or chunk.source_path,
        quote=quote,
    )
