"""Run the v0 eval dataset against the Answer pipeline.

Usage:
    uv run python eval/run_eval.py                    # all questions
    uv run python eval/run_eval.py --limit 3          # first 3 only
    uv run python eval/run_eval.py --only def-001     # one specific id
    uv run python eval/run_eval.py --dataset path     # use a different YAML

Outputs:
- Per-question trace JSONs into `traces/`
- One Markdown report into `eval/reports/eval_<version>_<timestamp>.md`
- Console summary

Auto-checks (deterministic, no LLM judge):
- answer_present
- all_inline_citations_have_entries
- all_citations_referenced_inline
- cited_urls_in_fetched_pages
- no_duplicate_citation_entries
- quotes_appear_in_sources
- expected_topics_mentioned       (skipped when no expected_topics)
- answer_disclaims_uncertainty    (only when expect_disclaimer: true)
- no_redundant_sentences          (probe — flags restated/duplicated claims)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

# Silence langgraph's transitive `allowed_objects` deprecation warning fired
# at import time from langgraph.cache.base. It's a langgraph-internal API
# change we can't influence from here; the warning leaks into eval output.
# Filter by the warning class (most reliable), with a message regex as
# belt-and-suspenders for any langchain-core variant that doesn't expose
# the class cleanly.
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message=r"The default value of .allowed_objects.")

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

# Make src/ importable when running this file directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from answer.pipeline import answer_question  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVAL_DIR / "dataset_v0.yaml"
REPORTS_DIR = EVAL_DIR / "reports"
TRACES_DIR = _PROJECT_ROOT / "traces"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    # For checks whose natural unit is "X of Y items" rather than a binary
    # pass/fail (e.g. quote-level verbatim across a citation group), set
    # `num` and `denom`. The report surfaces these directly as "X/Y" in the
    # per-question detail and aggregates them across rows in the pass-rate
    # table. `passed` is still set (True iff num == denom) so the row-level
    # "all checks passed" summary stays well-defined.
    num: int | None = None
    denom: int | None = None
    # Tag separating "correctness" (the answer is broken if this fails) from
    # "probe" (a diagnostic finding — a row can be perfectly correct AND
    # fail a probe). The row-level clean count uses correctness only; probe
    # results show as informational. Defaults to "correctness" so existing
    # check call sites stay unchanged.
    kind: str = "correctness"  # "correctness" | "probe"


# Phrases that indicate the model declined or flagged insufficient sources.
# Kept loose on purpose; we'd rather catch a real disclaimer than miss one.
#
# v1 expansion: the disclaim path uses first-person natural prose
# ("I couldn't find..."), so common contractions and natural negation
# patterns are also included. Without these, a perfectly correct
# natural-sounding disclaim would be marked as failing the check.
_DISCLAIMER_PHRASES = [
    "the sources do not",
    "the provided sources do not",
    "the available sources do not",
    "the given sources do not",
    "sources do not provide",
    "sources do not specify",
    "sources do not mention",
    "sources don't",
    "do not contain",
    "does not contain",
    "not enough information",
    "insufficient information",
    "no information",
    "no specific information",
    "no published",
    "not available",
    "cannot find",
    "could not find",
    "couldn't find",
    "can't find",
    "didn't find",
    "haven't found",
    "haven't been able",
    "i cannot",
    "i can't",
    "i couldn't",
    "i don't have",
    "i'm not able",
    "wasn't able",
    "weren't able",
    "unable to",
    "cannot help",
    "no relevant",
]


# --- helpers ----------------------------------------------------------------


def _to_jsonable(obj):
    """As run.py: drops `embedding` keys to keep traces lean."""
    if isinstance(obj, BaseModel):
        return _to_jsonable(obj.model_dump())
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {
            k: _to_jsonable(v) for k, v in obj.items() if k != "embedding"
        }
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text)[:n].strip("_")
    return s or "run"


def _save_trace(state: dict, qid: str, question: str) -> Path:
    TRACES_DIR.mkdir(exist_ok=True)
    fname = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{qid}_{_slug(question)}.json"
    )
    path = TRACES_DIR / fname
    path.write_text(
        json.dumps(_to_jsonable(state), indent=2, default=str), encoding="utf-8"
    )
    return path


_MD_NOISE = re.compile(r"[*_`#|>•·‣◦▪▫]+")
# Consume the marker plus any leading whitespace and trailing
# whitespace/punctuation so things like `Inc. [1].` collapse to `Inc.`
# instead of `Inc..` — which would fail a verbatim substring check
# against a source that just says `Inc. ...`.
_CITE_MARKER = re.compile(r"\s*\[\d+\][\s.,;:!?]*")
# Fraction of the (normalized) quote that must appear as one contiguous
# match somewhere in the (normalized) source to count as verbatim.
_VERBATIM_THRESHOLD = 0.80

# Two sentences in the same answer that share a long contiguous run of text
# are almost always a restatement of the same claim — the answer model
# summarizing the same fact from two sources in two sentences instead of
# merging them into one multi-cited claim. We flag a sentence pair when
# their normalized forms share a verbatim span of at least this many
# characters. 60 chars is ~10 words of contiguous overlap: long enough that
# coincidental collision between two genuinely distinct claims is negligible.
_REDUNDANT_SPAN_MIN = 60

# Split answer prose into sentences on end-punctuation followed by
# whitespace. Good enough for the model's clean prose; a rare mis-split only
# costs us one pairwise comparison, never correctness.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Just the bracket token `[1]` — used by the sentence splitter. Unlike
# `_CITE_MARKER` (which also consumes the trailing sentence period so quote
# checks match cleanly), this leaves surrounding punctuation intact so the
# sentence-ending `.` after `... [1].` survives to split on.
_CITE_BRACKET = re.compile(r"\[\d+\]")


def _norm(s: str) -> str:
    """Normalize text for substring comparison.

    Strips Markdown formatting markers, citation markers (`[1]`, `[12]`, ...),
    and ALL whitespace, then lowercases.

    Whitespace removal is whitespace-agnostic on purpose: crawl4ai's
    markdown extractor can drop spaces around `**bold**` markers, producing
    fused words like "thejacob t. schwartzprofessor" in the source we feed
    the model. The model writes clean spacing in its quote, which would
    fail a naïve substring check. Citation-marker removal handles a second
    artifact: the model often appends its own `[1]` to the quote text,
    which obviously isn't in the source. Both are model/scraper artifacts,
    not real differences, so the eval forgives them. Collisions are
    negligibly rare for quote-length strings.
    """
    s = _MD_NOISE.sub("", s)
    s = _CITE_MARKER.sub("", s)
    return re.sub(r"\s+", "", s).lower()


def _quote_in_text(quote: str, source: str) -> bool:
    """Pass if the longest contiguous chunk of the normalized quote that
    appears verbatim in the normalized source covers at least
    `_VERBATIM_THRESHOLD` of the quote's length.

    Strict substring matching is too brittle: the model often adds a small
    standalone-framing preamble like "RAG enhances..." or "With RAG, ..."
    to a quote that is otherwise verbatim. The longest-common-substring
    rule passes those (where the long verbatim core dominates the quote)
    while still failing real paraphrasing, clause omissions, and
    fabrications (where no long contiguous match exists).
    """
    nq = _norm(quote)
    ns = _norm(source)
    if not nq or not ns:
        return False
    if nq in ns:
        return True
    sm = SequenceMatcher(None, nq, ns, autojunk=False)
    match = sm.find_longest_match(0, len(nq), 0, len(ns))
    return match.size >= _VERBATIM_THRESHOLD * len(nq)


def _split_sentences(text: str) -> list[str]:
    """Split answer prose into sentences for pairwise redundancy checks.

    Citation brackets are stripped first (with `_CITE_BRACKET`, which
    preserves the sentence-ending period) so a trailing `[1]` vs `[2]`
    doesn't perturb the comparison between two otherwise-identical claims.
    """
    cleaned = _CITE_BRACKET.sub("", text)
    return [s.strip() for s in _SENTENCE_SPLIT.split(cleaned) if s.strip()]


def _find_redundant_sentence_pairs(
    text: str, *, span_min: int = _REDUNDANT_SPAN_MIN
) -> list[tuple[int, int, str]]:
    """Find sentence pairs that restate the same content.

    Returns `(i, j, shared_span)` for each pair of sentences whose
    normalized forms share a contiguous verbatim span of at least
    `span_min` characters. Reuses the same normalization + longest-common-
    substring approach as the quote-verbatim check (`_norm` +
    `SequenceMatcher.find_longest_match`): a 60+ char contiguous overlap
    between two different sentences is a strong copy-restate signal, while
    genuinely distinct claims essentially never collide that long.

    The returned `shared_span` is the normalized (whitespace-stripped,
    lowercased) overlap — enough to make the restatement legible in the
    report detail without reproducing the whole sentence.
    """
    normed = [_norm(s) for s in _split_sentences(text)]
    pairs: list[tuple[int, int, str]] = []
    for i, a in enumerate(normed):
        if len(a) < span_min:
            continue
        for j in range(i + 1, len(normed)):
            b = normed[j]
            if len(b) < span_min:
                continue
            m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
                0, len(a), 0, len(b)
            )
            if m.size >= span_min:
                pairs.append((i, j, a[m.a : m.a + m.size]))
    return pairs


# --- auto-checks ------------------------------------------------------------


def run_auto_checks(state: dict, entry: dict) -> list[CheckResult]:
    out: list[CheckResult] = []
    answer = state.get("final_answer")
    text = getattr(answer, "text", "") if answer else ""

    out.append(CheckResult("answer_present", bool(text), ""))
    if not text:
        return out

    citations = answer.citations
    inline_nums = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    cited_nums = {c.n for c in citations}

    missing = inline_nums - cited_nums
    out.append(
        CheckResult(
            "all_inline_citations_have_entries",
            not missing,
            f"missing entries for: {sorted(missing)}" if missing else "",
        )
    )

    unused = cited_nums - inline_nums
    out.append(
        CheckResult(
            "all_citations_referenced_inline",
            not unused,
            f"unused citation entries: {sorted(unused)}" if unused else "",
        )
    )

    pages: dict[str, str] = state.get("pages", {})
    bad_urls = [c.url for c in citations if c.url not in pages]
    out.append(
        CheckResult(
            "cited_urls_in_fetched_pages",
            not bad_urls,
            f"URLs cited but not fetched: {bad_urls}" if bad_urls else "",
        )
    )

    seen: set[int] = set()
    duplicate_ns: list[int] = []
    for c in citations:
        if c.n in seen:
            duplicate_ns.append(c.n)
        else:
            seen.add(c.n)
    out.append(
        CheckResult(
            "no_duplicate_citation_entries",
            not duplicate_ns,
            f"duplicate `n` values in citations list: {sorted(set(duplicate_ns))}"
            if duplicate_ns
            else "",
        )
    )

    # Each citation has a list of quotes (one per cited claim). Report as
    # a quote-level rate (num verbatim / total quotes) so a question with
    # 8/9 quotes verbatim isn't conflated with one where 0/9 pass. Detail
    # lists which quote indices failed (`n#index`).
    bad_quote_refs: list[str] = []
    total_quotes = 0
    for c in citations:
        page_text = pages.get(c.url, "")
        for i, quote in enumerate(c.quotes):
            total_quotes += 1
            if not _quote_in_text(quote, page_text):
                bad_quote_refs.append(f"{c.n}#{i}")
    verbatim_count = total_quotes - len(bad_quote_refs)
    out.append(
        CheckResult(
            "quotes_appear_in_sources",
            passed=not bad_quote_refs,
            detail=(
                f"quotes not found in source for: {bad_quote_refs}"
                if bad_quote_refs
                else ""
            ),
            num=verbatim_count,
            denom=total_quotes,
        )
    )

    expected = entry.get("expected_topics") or []
    if expected:
        text_lower = text.lower()
        hits = [t for t in expected if t.lower() in text_lower]
        missing_topics = sorted(set(expected) - set(hits))
        out.append(
            CheckResult(
                "expected_topics_mentioned",
                len(hits) == len(expected),
                f"hit {len(hits)}/{len(expected)}; missing: {missing_topics}"
                if missing_topics
                else "",
            )
        )

    if entry.get("expect_disclaimer"):
        text_lower = text.lower()
        found = any(p in text_lower for p in _DISCLAIMER_PHRASES)
        out.append(
            CheckResult(
                "answer_disclaims_uncertainty",
                found,
                "" if found else "expected a disclaimer phrase indicating insufficient sources",
            )
        )

    # --- Stress-test checks (fire only when their YAML field is present) ----
    # These extend the basic correctness checks above to probe the architecture's
    # agentic capabilities. Each one returns nothing when its YAML field is
    # absent, so the v0 dataset is unaffected.

    expected_qc = entry.get("expected_query_count")
    if expected_qc is not None:
        # Number of queries the planner emitted on the FINAL iteration is in
        # state["queries"]; the full multi-iteration history accumulates in
        # state["query_history"]. For this check we want what the planner
        # *initially* decided on, which on iter 1 is also state["queries"].
        # If the loop fires and iter 2 plans a different count, the final
        # state["queries"] reflects iter 2 — that's still meaningful but
        # different semantics; document if this ever bites us.
        actual_qs = state.get("queries") or []
        actual_count = len(actual_qs)
        out.append(
            CheckResult(
                "expected_query_count",
                actual_count == expected_qc,
                "" if actual_count == expected_qc
                else f"expected {expected_qc} queries, got {actual_count}: {actual_qs}",
                kind="probe",
            )
        )

    expected_min_sources = entry.get("expected_min_unique_sources")
    if expected_min_sources is not None:
        n_unique = len({c.url for c in citations}) if citations else 0
        out.append(
            CheckResult(
                "expected_min_unique_sources",
                n_unique >= expected_min_sources,
                "" if n_unique >= expected_min_sources
                else f"expected ≥{expected_min_sources} unique source URLs, got {n_unique}",
                kind="probe",
            )
        )

    expected_loop = entry.get("expected_loop_fired")
    if expected_loop is not None:
        # `state["iteration"]` counts COMPLETED loop iterations. iter == 1
        # means only the initial pass ran; iter ≥ 2 means the refinement
        # loop fired at least once.
        iterations = state.get("iteration", 0)
        fired = iterations > 1
        out.append(
            CheckResult(
                "expected_loop_fired",
                fired == expected_loop,
                "" if fired == expected_loop
                else (
                    f"expected loop_fired={expected_loop}, "
                    f"actual iterations={iterations} (fired={fired})"
                ),
                kind="probe",
            )
        )

    expected_min_disclaim_cites = entry.get("expected_min_citations_in_disclaim")
    if expected_min_disclaim_cites is not None:
        text_lower = text.lower()
        has_disclaim = any(p in text_lower for p in _DISCLAIMER_PHRASES)
        n_cites = len(citations)
        passed = has_disclaim and n_cites >= expected_min_disclaim_cites
        if passed:
            detail = ""
        elif not has_disclaim:
            detail = "expected disclaimer phrase + citation(s), but no disclaim found"
        else:
            detail = (
                f"disclaim found, but cited only {n_cites} adjacent fact(s) "
                f"(expected ≥{expected_min_disclaim_cites})"
            )
        out.append(
            CheckResult(
                "disclaim_with_citation",
                passed,
                detail,
                kind="probe",
            )
        )

    # --- Structural-coherence probe (universal — fires on every answered row) ---
    # Restated claims are the headline v1 coherence defect: the answer model
    # summarizes the same fact from two sources in two near-identical
    # sentences instead of merging them into one multi-cited claim (seen on
    # stress-multi-002 and stress-refine-003). This probe flags sentence
    # pairs sharing a long verbatim span. Diagnostic, not a correctness gate
    # — a redundant answer is lower quality but not "wrong" — so kind=probe.
    redundant_pairs = _find_redundant_sentence_pairs(text)
    if redundant_pairs:
        shown = "; ".join(
            f'sentences {i}&{j} share "{span[:50]}…"'
            if len(span) > 50
            else f'sentences {i}&{j} share "{span}"'
            for i, j, span in redundant_pairs[:3]
        )
        more = "" if len(redundant_pairs) <= 3 else f" (+{len(redundant_pairs) - 3} more)"
        detail = f"{len(redundant_pairs)} restated pair(s): {shown}{more}"
    else:
        detail = ""
    out.append(
        CheckResult(
            "no_redundant_sentences",
            passed=not redundant_pairs,
            detail=detail,
            kind="probe",
        )
    )

    return out


# --- report rendering -------------------------------------------------------


def _check_status(c: CheckResult) -> str:
    """Visible status string for one check in the per-question detail.

    Fraction-based checks (with `num`/`denom`) show as `X/Y`. Binary
    checks show as `PASS`/`FAIL`. Both forms appear in the same column
    so the report stays scannable.
    """
    if c.num is not None and c.denom is not None:
        return f"{c.num}/{c.denom}"
    return "PASS" if c.passed else "FAIL"


def _is_correctness(c: CheckResult) -> bool:
    """A check counts toward 'row clean' only if it's a correctness check.
    Probes are diagnostic — a row can be perfectly correct and fail a probe."""
    return getattr(c, "kind", "correctness") == "correctness"


def render_report(rows: list[dict], dataset_version: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    by_cat: dict[str, dict] = {}
    # For fraction checks, also accumulate item-level num/denom across rows
    # so the pass-rate table can show "X/Y verbatim" instead of "K/T rows".
    check_stats: dict[str, dict] = {}
    for row in rows:
        cat = row["category"]
        d = by_cat.setdefault(
            cat, {"total": 0, "correctness_clean": 0, "all_clean": 0}
        )
        d["total"] += 1
        if row.get("error"):
            continue
        correctness_checks = [c for c in row["checks"] if _is_correctness(c)]
        if all(c.passed for c in correctness_checks):
            d["correctness_clean"] += 1
        if all(c.passed for c in row["checks"]):
            d["all_clean"] += 1
        for c in row["checks"]:
            cs = check_stats.setdefault(
                c.name,
                {
                    "pass": 0,
                    "total": 0,
                    "num": 0,
                    "denom": 0,
                    "fractional": False,
                    "kind": getattr(c, "kind", "correctness"),
                },
            )
            cs["total"] += 1
            if c.passed:
                cs["pass"] += 1
            if c.num is not None and c.denom is not None:
                cs["fractional"] = True
                cs["num"] += c.num
                cs["denom"] += c.denom

    n_total = len(rows)
    n_err = sum(1 for r in rows if r.get("error"))
    # "Correctness clean" — the row's answer is structurally valid AND
    # passes all correctness checks. This is the headline number.
    n_correctness_clean = sum(
        1
        for r in rows
        if not r.get("error")
        and all(c.passed for c in r["checks"] if _is_correctness(c))
    )
    # "All checks clean" — also passes every probe. Probes are diagnostic
    # findings, not pass/fail correctness, so this number is informational.
    n_all_clean = sum(
        1
        for r in rows
        if not r.get("error") and all(c.passed for c in r["checks"])
    )
    n_probe_checks = sum(
        1
        for r in rows
        if not r.get("error")
        for c in r["checks"]
        if not _is_correctness(c)
    )

    lines: list[str] = []
    lines.append(f"# Eval report — dataset {dataset_version}")
    lines.append("")
    lines.append(f"- Generated: `{now}`")
    lines.append(f"- Total questions: {n_total}")
    lines.append(
        f"- **Correctness clean: {n_correctness_clean}/{n_total}** — rows whose answer is structurally valid and passes every correctness check"
    )
    if n_probe_checks:
        n_probe_pass = sum(
            stats["pass"]
            for stats in check_stats.values()
            if stats["kind"] != "correctness"
        )
        n_probe_total = sum(
            stats["total"]
            for stats in check_stats.values()
            if stats["kind"] != "correctness"
        )
        lines.append(
            f"- Probe checks passed: {n_probe_pass}/{n_probe_total} — "
            "diagnostic findings about agentic behaviors; a probe failure "
            "does not mean the answer is wrong"
        )
        lines.append(
            f"- All checks (correctness + probes) clean: {n_all_clean}/{n_total} — informational only"
        )
    if n_err:
        lines.append(f"- Errors: {n_err}")
    lines.append("")

    lines.append("## Summary by category")
    lines.append("")
    if n_probe_checks:
        lines.append("| Category | Total | Correctness clean | All clean (incl. probes) |")
        lines.append("|---|---|---|---|")
        for cat, d in sorted(by_cat.items()):
            lines.append(
                f"| {cat} | {d['total']} | "
                f"{d['correctness_clean']}/{d['total']} | "
                f"{d['all_clean']}/{d['total']} |"
            )
    else:
        lines.append("| Category | Total | Correctness clean |")
        lines.append("|---|---|---|")
        for cat, d in sorted(by_cat.items()):
            lines.append(
                f"| {cat} | {d['total']} | {d['correctness_clean']}/{d['total']} |"
            )
    lines.append("")

    # Pass rate by check. If any probes exist, split into two tables for
    # readability. Otherwise (v0 dataset), one table.
    correctness_stats = sorted(
        (n, s) for n, s in check_stats.items() if s["kind"] == "correctness"
    )
    probe_stats = sorted(
        (n, s) for n, s in check_stats.items() if s["kind"] != "correctness"
    )

    def _row_for_check(name: str, d: dict) -> str:
        if d["fractional"]:
            return (
                f"| `{name}` | {d['num']}/{d['denom']} items "
                f"(rows fully clean: {d['pass']}/{d['total']}) |"
            )
        return f"| `{name}` | {d['pass']}/{d['total']} |"

    lines.append("## Pass rate by correctness check")
    lines.append("")
    lines.append("| Check | Pass rate |")
    lines.append("|---|---|")
    for name, d in correctness_stats:
        lines.append(_row_for_check(name, d))
    lines.append("")

    if probe_stats:
        lines.append("## Probe findings (informational, not pass/fail)")
        lines.append("")
        lines.append("| Probe | Pass rate |")
        lines.append("|---|---|")
        for name, d in probe_stats:
            lines.append(_row_for_check(name, d))
        lines.append("")

    lines.append("## Per-question detail")
    for row in rows:
        lines.append("")
        lines.append(f"### `{row['id']}` — {row['question']}")
        lines.append("")
        lines.append(f"- **Category:** {row['category']}")
        if row.get("error"):
            lines.append("- **Error:**")
            lines.append("")
            lines.append("```")
            lines.append(row["error"].rstrip())
            lines.append("```")
            continue
        # Group correctness checks first, then probes — visually delineates them.
        correctness_checks = [c for c in row["checks"] if _is_correctness(c)]
        probe_checks = [c for c in row["checks"] if not _is_correctness(c)]
        lines.append("- **Correctness checks:**")
        for c in correctness_checks:
            extra = f" — {c.detail}" if c.detail and not c.passed else ""
            lines.append(f"  - **[{_check_status(c)}]** `{c.name}`{extra}")
        if probe_checks:
            lines.append("- **Probe findings:**")
            for c in probe_checks:
                extra = f" — {c.detail}" if c.detail and not c.passed else ""
                lines.append(f"  - **[{_check_status(c)}]** `{c.name}`{extra}")
        lines.append("- **Answer:**")
        lines.append("")
        for ln in row["answer_text"].splitlines() or [""]:
            lines.append(f"  > {ln}")
        lines.append("")
        if row["citations"]:
            lines.append("- **Citations:**")
            for c in row["citations"]:
                lines.append(f"  - [{c['n']}] {c['url']}")
        if row.get("trace_path"):
            lines.append(f"- **Trace:** `{row['trace_path']}`")

    return "\n".join(lines) + "\n"


# --- runner -----------------------------------------------------------------


async def _run_one(entry: dict) -> dict:
    qid = entry["id"]
    question = entry["question"]
    print(f"\n{'#' * 60}")
    print(f"# {qid} — {question}")
    print(f"{'#' * 60}")

    try:
        state = await answer_question(question)
    except Exception:
        return {
            "id": qid,
            "category": entry.get("category", "uncategorized"),
            "question": question,
            "error": traceback.format_exc(limit=6),
            "checks": [],
            "answer_text": "",
            "citations": [],
        }

    trace_path = _save_trace(state, qid, question)
    checks = run_auto_checks(state, entry)
    answer = state.get("final_answer")
    return {
        "id": qid,
        "category": entry.get("category", "uncategorized"),
        "question": question,
        "checks": checks,
        "answer_text": getattr(answer, "text", "") or "",
        "citations": [c.model_dump() for c in (answer.citations if answer else [])],
        "trace_path": str(trace_path.relative_to(_PROJECT_ROOT)),
    }


async def _amain(args) -> None:
    dataset_path = Path(args.dataset)
    with dataset_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    version = data.get("version", "v0")
    questions: list[dict] = data["questions"]

    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        questions = [q for q in questions if q["id"] in wanted]
        if not questions:
            print(f"no question(s) matching {sorted(wanted)}")
            sys.exit(2)
    if args.limit:
        questions = questions[: args.limit]

    rows: list[dict] = []
    for entry in questions:
        rows.append(await _run_one(entry))

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = (
        REPORTS_DIR
        / f"eval_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    report_path.write_text(render_report(rows, version), encoding="utf-8")

    n_total = len(rows)
    n_err = sum(1 for r in rows if r.get("error"))
    n_correctness_clean = sum(
        1
        for r in rows
        if not r.get("error")
        and all(
            c.passed for c in r["checks"]
            if getattr(c, "kind", "correctness") == "correctness"
        )
    )
    print()
    print("=" * 60)
    print(
        f"eval done: {n_correctness_clean}/{n_total} correctness-clean, "
        f"{n_err} errored"
    )
    print(f"report: {report_path.relative_to(_PROJECT_ROOT)}")
    print("=" * 60)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help="path to the dataset YAML",
    )
    parser.add_argument("--limit", type=int, default=None, help="run only first N questions")
    parser.add_argument("--only", type=str, default=None, help="run specific question id(s), comma-separated")
    args = parser.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
