"""Markdown-aware chunking with deterministic, content-addressable chunk IDs.

A page's extracted markdown becomes a list of `Chunk`s — each one is an
atomic evidence unit that:

- is small enough (target ~400 tokens) that its embedding represents one
  idea sharply, not a smear across several;
- carries its source URL bound by ID, so the model can never misattribute
  a quote to the wrong page (the misattribution failure mode in v0);
- has a deterministic ID derived from `(url, position, content)` so that
  re-fetching the same page never duplicates entries in the vector store.

We use a character-based recursive splitter from `langchain_text_splitters`
(`RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN, ...)`)
because it understands markdown structure: it splits on headers, paragraphs,
list items, code blocks — in that order of preference — falling back to
character-level only when no semantic boundary fits. Token counting is
approximate (≈ 4 chars/token); we sidestep tiktoken to keep the dep tree
lean. If chunk-size tuning ever needs token-exact counts, swap the length
function — the interface stays the same.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from urllib.parse import urlparse

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS


@dataclass(frozen=True)
class Chunk:
    """One pre-embedding atomic evidence unit from a fetched page.

    `id` is deterministic over `(url, position, content_hash)`. Re-running
    over the same content yields the same ID, which is what makes
    "skip-if-already-embedded" cheap and correct in the vector store.
    """

    id: str
    text: str
    url: str
    domain: str  # bare `example.com`, what the LLM sees in the prompt
    position: int  # 0-based index of this chunk within its source page


def _domain_of(url: str) -> str:
    """Bare domain, no leading `www.`. What the answer prompt shows the model."""
    netloc = urlparse(url).netloc
    return netloc.removeprefix("www.") if netloc else url


def _chunk_id(url: str, position: int, content: str) -> str:
    """Deterministic 16-char chunk ID over (url, position, content-hash).

    16 hex chars = 64 bits of entropy. Collision risk over realistic corpora
    (10⁶ chunks) is ~10⁻⁸ — fine. This ID is the STORAGE identity (Chroma
    document id), enabling content-addressable upsert dedup across runs
    and persistent corpora. The LLM never sees this string directly — the
    answer/critique prompts surface chunks as small prompt-local integers
    (1..N) for token efficiency and model copy-reliability; Python maps
    those local integers back to the storage hex at render time. See
    `prompts._format_chunk` for the prompt-layer story.

    Including `position` means two pages with identical text at different
    positions still get distinct IDs (rare but possible — boilerplate
    paragraphs). Including the content hash means the same page re-fetched
    after a content change produces a different ID and thus a fresh embed —
    the right behavior for time-sensitive sources.
    """
    h = hashlib.sha1(
        f"{url}|{position}|{hashlib.sha1(content.encode('utf-8')).hexdigest()}".encode("utf-8")
    ).hexdigest()
    return h[:16]


# Module-level splitter — built once, reused. The Language.MARKDOWN preset
# uses separators ["\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ",
# "```\n", "\n\n***\n\n", ..., "\n\n", "\n", " ", ""] which prefers semantic
# boundaries (headers first, paragraphs next, sentences/words last).
_SPLITTER = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=CHUNK_SIZE_CHARS,
    chunk_overlap=CHUNK_OVERLAP_CHARS,
    length_function=len,
)


def chunk_page(url: str, markdown: str) -> list[Chunk]:
    """Split one fetched page into ordered, ID-tagged chunks.

    Empty / whitespace-only input returns []. The pipeline already drops
    empty pages before chunking, but this is defensive — an unfetched URL
    shouldn't become a zombie chunk.
    """
    if not markdown or not markdown.strip():
        return []
    domain = _domain_of(url)
    texts = _SPLITTER.split_text(markdown)
    return [
        Chunk(
            id=_chunk_id(url, i, t),
            text=t,
            url=url,
            domain=domain,
            position=i,
        )
        for i, t in enumerate(texts)
    ]


def chunk_pages(pages: dict[str, str]) -> list[Chunk]:
    """Chunk every page in `pages` and return a single flat list.

    Order is stable: pages are processed in `dict.items()` order (insertion
    order since 3.7), then chunks within each page by position. Stable
    ordering matters for trace JSON determinism and for the answer prompt's
    "group by source URL" rendering.
    """
    out: list[Chunk] = []
    for url, md in pages.items():
        out.extend(chunk_page(url, md))
    return out
