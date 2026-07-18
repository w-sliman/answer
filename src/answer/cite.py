"""Citation assignment AFTER answer generation (draft-then-cite, step 3).

The answer stage now drafts freeform (no inline citations), then an attribution
pass maps each factual claim back to its supporting chunk(s). This module takes
that attribution result and the freeform draft and produces the final cited
answer by **annotating the draft in place**: the draft is kept VERBATIM and
`[n]` source markers are appended to the line each supported claim came from,
located by deterministic containment matching (no model, no embeddings).

Lifted from the validated eval prototype (`annotate_inplace_probe` /
`draft_then_cite_playground`, 96% clean placement, 0 ambiguous) with ONE fix:
all markers landing on a line are merged into a SINGLE sorted, de-duplicated
group (`[1][2]`) instead of the prototype's space-joined per-claim groups
(`[1] [1][2]`), which read noisily. `[n]` numbering is per unique URL, first-seen
in claim order — the same contract downstream (`run.py`, the report) expects.
"""
from __future__ import annotations

import re

from .prompts import Citation, PlacedSource, Placement

_MD = re.compile(r"[*_`>#|]+")
_STOP = {"the", "and", "for", "with", "that", "this", "are", "its", "has",
         "was", "which", "from", "into", "can", "not", "but", "also"}
#: containment below this: the claim's content isn't clearly in any one line,
#: so its marker is left off rather than attached to a wrong line.
_PLACE_MIN = 0.55


def _clean_quote(q: str) -> str:
    """Strip extraction artifacts from an attributed quote so it reads as a quote.

    The attribution model copies quotes verbatim out of markdown-ish chunk text,
    so they arrive carrying junk: a leading `::` prefix, the SOURCE's own inline
    `[n]` citation markers, markdown emphasis/heading/quote marks, and table pipes.
    None of that belongs in a citation shown to a reader. Conservative — it only
    removes formatting, never words.
    """
    s = q.strip()
    s = re.sub(r"^:+\s*", "", s)          # leading "::" extraction prefix
    s = re.sub(r"\[\d+\]", "", s)          # the source's OWN inline [n] citations
    s = re.sub(r"[*`#>]", "", s)           # markdown bold / italic / code / heading / quote
    s = re.sub(r"\s*\|\s*", " · ", s)      # table pipes -> middot separators
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" ·")


def _toks(s: str) -> set[str]:
    s = _MD.sub(" ", s).lower()
    return {t for t in re.findall(r"[a-z0-9]+", s) if len(t) > 3 and t not in _STOP}


def _cov(claim: set[str], unit: set[str]) -> float:
    """Containment: fraction of the CLAIM's tokens covered by the unit. A claim
    is derived from a draft unit, so the right question is 'is the claim's
    content in this unit', not symmetric similarity."""
    if not claim:
        return 0.0
    return len(claim & unit) / len(claim)


def _annotatable_lines(draft: str) -> list[tuple[int, str]]:
    """(index, line) for lines a marker could attach to — skip blanks and pure
    section headers (`### Title`, `**Bold label:**`) so markers land on content."""
    out = []
    for i, line in enumerate(draft.splitlines()):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if re.fullmatch(r"\*\*[^*]+:\*\*", s):
            continue
        if len(_toks(s)) < 3:
            continue
        out.append((i, line))
    return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _best_match(claim: str, candidates: list[str]) -> tuple[int, float]:
    """(best_idx, best_containment) — how well the claim's content is contained
    in the best-matching candidate unit."""
    ct = _toks(claim)
    best_i, best_s = -1, 0.0
    for i, c in enumerate(candidates):
        s = _cov(ct, _toks(c))
        if s > best_s:
            best_i, best_s = i, s
    return best_i, best_s


def annotate_in_place(
    draft: str, attribution, hits,
) -> tuple[str, list[Citation], list[Placement]]:
    """Annotate the freeform `draft` with `[n]` markers from an `AttributionResult`.

    `hits` are the source chunks (1-indexed to match `chunk_id`). Returns:
      - the annotated draft (structure preserved),
      - the flattened per-URL `Citation` bibliography (each source's aggregated
        quotes — the "sources used" list), and
      - `Placement`s: one per annotated sentence, each carrying, PER MARKER, the
        SPECIFIC quote(s) that back THAT sentence. This is the per-claim evidence
        — it is what makes `[n]` on a line mean "this exact quote supports this
        exact sentence", rather than the source's whole quote bag (a per-URL
        grouping showed, e.g., LangChain quotes under a Haystack sentence).

    Unsupported claims carry no marker (the grounding loop upstream is meant to
    have removed them; a terminal draft that still holds one keeps the sentence
    unmarked rather than mis-citing it). A supported claim whose content can't be
    located on any line (containment < `_PLACE_MIN`) is left unmarked rather than
    attached to the wrong line.

    `[n]` numbering is unchanged (per unique URL, first-seen in claim order) and
    the rendered text is byte-identical to the pre-placement version — only the
    per-sentence evidence is newly surfaced.
    """
    # 1. first-seen URL -> n, per-URL aggregated quotes (bibliography), and each
    #    supported claim resolved to its concrete (n, url, cleaned-quotes) sources.
    url_to_n: dict[str, int] = {}
    quotes_by_url: dict[str, list[str]] = {}
    resolved: list[tuple[str, list[tuple[int, str, list[str]]]]] = []
    for c in attribution.claims:
        if not c.supported:
            continue
        sources: list[tuple[int, str, list[str]]] = []
        for cit in c.citations:
            if 1 <= cit.chunk_id <= len(hits):
                hit = hits[cit.chunk_id - 1]
                url = hit.url or hit.domain
                n = url_to_n.setdefault(url, len(url_to_n) + 1)
                cleaned = [cq for q in cit.quotes if (cq := _clean_quote(q))]
                sources.append((n, url, cleaned))
                for q in cleaned:
                    if q not in quotes_by_url.setdefault(url, []):
                        quotes_by_url[url].append(q)
        if sources:
            resolved.append((c.claim, sources))

    # 2. sentence-level candidates, each tagged with the line it lives on
    lines = draft.splitlines()
    cand: list[tuple[int, str]] = []
    for idx, line in _annotatable_lines(draft):
        for sent in _sentences(line) or [line]:
            cand.append((idx, sent))
    cand_texts = [s for _, s in cand]

    # 3. place each claim on its best-matching line; accumulate BOTH the marker
    #    set (for rendering) and the per-marker quotes (for the placements).
    per_line: dict[int, set[int]] = {}
    per_line_q: dict[int, dict[int, dict]] = {}  # line -> n -> {url, quotes:[...]}
    for claim, sources in resolved:
        bi, bs = _best_match(claim, cand_texts)
        if bi < 0 or bs < _PLACE_MIN:
            continue
        line = cand[bi][0]
        for n, url, quotes in sources:
            per_line.setdefault(line, set()).add(n)
            slot = per_line_q.setdefault(line, {}).setdefault(n, {"url": url, "quotes": []})
            for q in quotes:
                if q not in slot["quotes"]:
                    slot["quotes"].append(q)

    # 4. render: draft verbatim, ONE merged sorted marker group per line
    out: list[str] = []
    for i, line in enumerate(lines):
        if i in per_line:
            marker = "".join(f"[{n}]" for n in sorted(per_line[i]))
            out.append(f"{line} {marker}")
        else:
            out.append(line)

    citations = [
        Citation(n=n, url=url, quotes=quotes_by_url.get(url, []))
        for url, n in sorted(url_to_n.items(), key=lambda kv: kv[1])
    ]
    placements = [
        Placement(
            line=lines[i].strip(),
            markers=sorted(per_line[i]),
            evidence=[
                PlacedSource(n=n, url=per_line_q[i][n]["url"],
                             quotes=per_line_q[i][n]["quotes"])
                for n in sorted(per_line_q.get(i, {}))
            ],
        )
        for i in sorted(per_line)
    ]
    return "\n".join(out), citations, placements
