"""V1 pipeline as a LangGraph StateGraph.

Phase 4 topology (loop closed via sufficiency check):

    generate_queries  ←──────────────────────┐
        -> search                            │
        -> pick_links  (excludes fetched)    │
        -> fetch       (accumulates pages)   │
        -> chunk_and_index                   │
        -> retrieve    (top-K=40 from Chroma)│
        -> rerank      (cross-encoder)       │
        -> select_chunks (MMR top-N=8)       │
        -> sufficiency_check                 │
            │                                │
            ├── insufficient AND under budget─┘
            │
            └── sufficient | out of budget | diminishing returns
                -> answer
                -> END

Adaptive loop. After the selection stage, `sufficiency_check` decides
whether the chunk pool is strong enough to answer over (score-based
floors). If it isn't AND we still have budget AND the last pick_links
returned new URLs, we loop back to `generate_queries` for a refinement
pass. The refinement prompt sees the prior queries (`query_history`)
and instructs the planner to take a different angle.

Termination guarantees:
1. `sufficient = True` — we stop.
2. `iteration >= MAX_ITERATIONS` — we stop.
3. `picked_urls == []` (no new URLs to fetch) — we stop.

(1) is the happy path. (2) is the budget cap. (3) is the
diminishing-returns exit — if every URL ddgs surfaces is already in
`fetched_urls`, looping would just re-fetch the same pages.

State accumulators:
- `query_history`: every query ever planned in this run (for
  refinement-prompt context).
- `fetched_urls`: every URL ever fetched (for pick_links exclusion).
- `pages`: every page ever fetched (chunked once per URL via Chroma's
  content-addressable upsert).

Multi-query fan-out (Phase 3). The planner emits 1 query for factoid
questions, 2 for comparisons, 3 for genuinely multi-faceted questions.
Model decides the count under a hard schema cap (1..3).

Two-stage retrieval (Phase 2). Dense embedding finds K candidates;
cross-encoder rerank sharpens; MMR picks top-N balancing relevance
against diversity.

Structural citation safety (Phase 1). Each retrieved chunk has a
deterministic content-addressable id; the answer model cites by chunk_id
and Python resolves chunk_id → URL deterministically. The model cannot
misattribute a quote to the wrong URL because it never writes URLs.
"""
from __future__ import annotations

import time
import uuid
from typing import Callable, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from .chunking import Chunk, chunk_pages
from .config import (
    MAX_CRITIQUE_ATTEMPTS,
    MAX_ITERATIONS,
    MMR_LAMBDA,
    N_LINKS,
    TOP_K_RETRIEVAL,
    TOP_N_ANSWER,
)
from .embedding import embed_query
from .fetch import fetch_pages
from .llm import get_llm, with_schema
from .mmr import select_mmr
from .cite import annotate_in_place
from .prompts import (
    Answer,
    AttributionResult,
    LinkPicks,
    PlannedQueries,
    RefinedQueries,
    build_answer_messages_freeform,
    build_attribution_messages,
    build_link_picker_messages,
    build_query_generation_messages_needs,
)
from .rerank import rerank
from .search import SearchResult, search
from .sufficiency import SufficiencyResult, judge_sufficiency
from .vector_store import (
    RetrievalHit,
    get_collection,
    similarity_search,
    upsert_chunks,
)


# --- Render helpers ----------------------------------------------------------


def _score_for_ordering(h: RetrievalHit) -> float:
    """Best available relevance score for prompt ordering.

    Prefer rerank_score (sharper signal from the cross-encoder); fall
    back to retrieval_score when no rerank has run. This keeps the same
    grouping function working across phases — Phase 1 (no rerank) and
    Phase 2+ (rerank) use the same code path.
    """
    return h.rerank_score if h.rerank_score is not None else h.retrieval_score


def _group_chunks_for_prompt(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """Reorder retrieved hits so chunks from the same URL appear together,
    score-descending within each group. Groups themselves are ordered by
    each URL's BEST hit (most-relevant source first).

    Why grouping helps: the answer model writes more coherent prose when
    it can discuss one source at a time, rather than zig-zagging between
    different domains. We measured the symptom in v0 as occasional
    paragraph-level incoherence on comparative questions.
    """
    by_url: dict[str, list[RetrievalHit]] = {}
    for h in hits:
        by_url.setdefault(h.url, []).append(h)
    for hs in by_url.values():
        hs.sort(key=_score_for_ordering, reverse=True)
    url_order = sorted(
        by_url.keys(),
        key=lambda u: _score_for_ordering(by_url[u][0]),
        reverse=True,
    )
    out: list[RetrievalHit] = []
    for u in url_order:
        out.extend(by_url[u])
    return out


# --- State -------------------------------------------------------------------


class AnswerState(TypedDict):
    """Shared state threaded through the graph.

    `job_id` namespaces the Chroma collection. For v1.0 ad-hoc questions a
    fresh uuid per call keeps eval isolation automatic; the longer-term
    competitive-analysis system will reuse the same `job_id` across many
    questions in a research session so chunks accumulate.
    """

    question: str
    job_id: str

    # --- Loop control (Phase 4) ---
    # `iteration` counts COMPLETED loop iterations (bumped at the end of
    # each sufficiency check). 0 on entry; max value = MAX_ITERATIONS.
    iteration: NotRequired[int]
    max_iterations: NotRequired[int]
    # Set by a refinement pass (iteration > 0) when the planner judges the
    # gap unfillable by any search (`RefinedQueries.searchable == False`) or
    # produces no usable new query. `route_after_generate` reads it to skip
    # the doomed search and go straight to `answer` (the disclaim path).
    refine_stop: NotRequired[bool]
    # Accumulators (NOT overwritten per iteration — manually merged in
    # the nodes that produce them).
    query_history: NotRequired[list[str]]
    fetched_urls: NotRequired[list[str]]

    # --- Per-iteration (overwritten each loop pass) ---
    queries: NotRequired[list[str]]
    query_reasoning: NotRequired[str]
    search_results: NotRequired[list[SearchResult]]
    picked_urls: NotRequired[list[str]]
    # `pages` accumulates across iterations — the answer node and eval
    # checks both need the full picture of every URL we've fetched in
    # this run.
    pages: NotRequired[dict[str, str]]
    new_chunks: NotRequired[list[Chunk]]
    retrieved: NotRequired[list[RetrievalHit]]
    selected: NotRequired[list[RetrievalHit]]
    sufficiency: NotRequired[SufficiencyResult]
    # --- Answer node (draft + ground) output ---
    # The freeform grounded draft, BEFORE citation. `run_answer_stage` runs the
    # draft → attribute → regenerate loop and hands the terminal draft +
    # attribution to the `cite` node. Kept in state so `cite` (and traces) can
    # read the answer as generated, separate from the cited render.
    draft: NotRequired[str]
    attribution: NotRequired[AttributionResult]
    answer_attempts: NotRequired[int]

    # --- Cite node output ---
    # The grounded draft annotated in place with `[n]` markers (deterministic,
    # model-free). Produced by the `cite` node from `draft` + `attribution`.
    final_answer: NotRequired[Answer]


#: Per-stage status sink. `print` by default (CLI use); the web UI swaps this
#: for a callback that streams lines to the browser. See `set_status_handler`.
_status_handler: Callable[[str], None] = print


def set_status_handler(handler: Callable[[str], None] | None) -> None:
    """Swap the per-stage status messenger.

    Pass a callable to redirect output (e.g. for web-UI
    streaming). Pass `None` to reset to the default `print`.

    The handler receives one already-formatted status line at a time —
    the strings the nodes pass to `_status`. Stage tags like `[plan]`,
    `[search]`, etc. are part of the message; downstream consumers can
    parse them to drive structured rendering.

    Caveat: this is module-global, so concurrent pipeline runs in the
    same process will share the handler. Fine for single-session demos
    (Colab + one browser); a multi-tenant deployment would need per-request
    handler scoping (e.g. via contextvars).
    """
    global _status_handler
    _status_handler = handler if handler is not None else print


def _status(msg: str) -> None:
    """Concise one-line status update for the terminal watcher.

    Pipeline nodes call this to report what just happened. Stage tags in
    square brackets at the start make output scannable. Full state detail
    (prompts, scored candidate lists, embeddings) lives in the trace JSON,
    not on stdout — this is purely for the human watching the run.

    The actual sink is `_status_handler` — `print` by default, swappable
    via `set_status_handler` for non-CLI consumers (e.g. the web UI).
    """
    _status_handler(msg)


# --- Structured event channel (for rich UIs) --------------------------------
# Separate from `_status` (flat terminal strings): nodes emit structured events
# a UI can render as timed, expandable, per-item sections. `_instrument` wraps
# each graph node to emit stage_start / stage_end (with elapsed timing)
# automatically, so nodes only need to emit their own DATA payloads via
# `_event("stage_data", stage=..., ...)`. Backward-compatible — the string
# `_status` stream is untouched; a consumer can use either or both.

_event_handler: Callable[[dict], None] | None = None


def set_event_handler(handler: Callable[[dict], None] | None) -> None:
    """Install a structured-event sink (or None to disable). Complements
    `set_status_handler`; a UI can consume both channels."""
    global _event_handler
    _event_handler = handler


def _event(kind: str, **data) -> None:
    """Emit one structured event. Silently no-ops when no sink is installed,
    and never lets a misbehaving sink break the pipeline run."""
    if _event_handler is not None:
        try:
            _event_handler({"kind": kind, "t": time.perf_counter(), **data})
        except Exception:  # noqa: BLE001 — a UI sink must never break the run
            pass


def _instrument(name: str, fn):
    """Wrap a node coroutine to emit stage_start / stage_end with elapsed s."""

    async def wrapped(state):
        _event("stage_start", stage=name)
        t0 = time.perf_counter()
        try:
            return await fn(state)
        finally:
            _event("stage_end", stage=name, elapsed=time.perf_counter() - t0)

    wrapped.__name__ = getattr(fn, "__name__", name)
    return wrapped


# --- Nodes -------------------------------------------------------------------


async def generate_queries_node(state: AnswerState) -> dict:
    """LLM call: plan search queries optimized for ddgs.

    Uses the `needs` planner (locked in): decompose
    by the distinct pieces of information the question needs, one query per
    piece, non-redundant, fewer queries are better. No in-prompt schema
    (redundant given `with_structured_output`); structured via
    `PlannedQueries` (one justification per query).

    Loop-aware. On the FIRST pass (iteration 0) it uses the locked first-attempt
    prompt bound to `PlannedQueries` — unchanged behavior. On a REFINEMENT pass
    (iteration > 0, driven by an insufficient sufficiency verdict) it feeds the
    prior queries + the judge's reason into the refinement block and binds
    `RefinedQueries`, which can either return NEW gap-targeting queries or set
    `searchable=False` to STOP (the gap is unfillable by any search). A stop —
    or a refinement that yields no usable new query — sets `refine_stop` so
    `route_after_generate` skips the doomed search and disclaims over what we
    already have.
    """
    history = list(state.get("query_history") or [])
    iteration_in = state.get("iteration", 0)
    suf = state.get("sufficiency")
    refining = iteration_in > 0 and suf is not None

    llm = get_llm()

    if not refining:
        structured_llm = with_schema(llm, PlannedQueries)
        messages = build_query_generation_messages_needs(state["question"])
        result: PlannedQueries = await structured_llm.ainvoke(messages)
        planned = result.queries
        searchable = True
    else:
        _status(f"\n--- Refining (iteration {iteration_in + 1}) ---")
        structured_llm = with_schema(llm, RefinedQueries)
        messages = build_query_generation_messages_needs(
            state["question"],
            prior_queries=history,
            sufficiency_reason=suf.reason,
        )
        refined: RefinedQueries = await structured_llm.ainvoke(messages)
        planned = refined.queries
        searchable = refined.searchable
        if not searchable:
            _status(f"[plan] refinement STOP (unsearchable): {refined.reason}")

    queries = [pq.query.strip() for pq in planned if pq.query.strip()]

    # Stop the loop when a refinement pass judged the gap unfillable, or when it
    # produced no usable new query (searching again would just repeat / no-op).
    if refining and (not searchable or not queries):
        _event(
            "stage_data", stage="generate_queries",
            iteration=iteration_in + 1, refine_stop=True,
            searchable=searchable, queries=[],
        )
        return {"refine_stop": True}

    # PlannedQuery carries its own `why` (no single overall reasoning) — join
    # them so the trace's query_reasoning field stays meaningful.
    reasoning = "; ".join(f"{pq.query}: {pq.why}" for pq in planned)
    label = "query" if len(queries) == 1 else "queries"
    _status(f"[plan] {len(queries)} {label}: {queries}")
    _status(f"[plan] reasoning: {reasoning}")
    _event(
        "stage_data",
        stage="generate_queries",
        iteration=iteration_in + 1,
        queries=[{"query": pq.query.strip(), "why": pq.why}
                 for pq in planned if pq.query.strip()],
    )
    return {
        "queries": queries,
        "query_reasoning": reasoning,
        # Accumulator: the prior history plus this iteration's new queries.
        "query_history": history + queries,
    }


async def search_node(state: AnswerState) -> dict:
    """Run every planned query through ddgs, merge results, dedup by URL.

    Why dedup by URL preserving first-seen order:
    - Two queries about the same topic often return overlapping pages
      (especially authoritative sources like wikipedia / official docs).
    - The duplicate URLs are *signal*, not noise — a page that surfaces
      across multiple queries is broadly relevant. But we only want to
      fetch it once and let retrieval decide its weight.
    - First-seen order means the search ordering bias is mild: a page
      that ranked #1 in query 1 keeps that ordering, while a page that
      only appears in query 2 lands later in the merged list. Reasonable
      heuristic without imposing a more complex re-ranking.
    """
    queries = state.get("queries") or []
    all_results: list[SearchResult] = []
    seen_urls: set[str] = set()
    for i, q in enumerate(queries):
        _event("search_query_start", index=i, query=q)
        qt0 = time.perf_counter()
        results = search(q, max_results=10)
        _event(
            "search_query_end",
            index=i,
            query=q,
            elapsed=time.perf_counter() - qt0,
            n=len(results),
            results=[{"title": r.title, "url": r.url, "snippet": r.snippet}
                     for r in results],
        )
        for r in results:
            if r.url and r.url not in seen_urls:
                seen_urls.add(r.url)
                all_results.append(r)
    _status(
        f"[search] {len(queries)} {'query' if len(queries) == 1 else 'queries'}"
        f" → {len(all_results)} unique URLs"
    )
    _event("stage_data", stage="search", n_queries=len(queries),
           n_unique=len(all_results))
    return {"search_results": all_results}


async def pick_links_node(state: AnswerState) -> dict:
    """LLM call: choose URLs worth fetching, with reasons.

    Phase 3: widened from 3 → N_LINKS (10). With multi-query fan-out
    producing 10-30 unique candidate URLs, picking 10 gives retrieval a
    much bigger pool to curate from than v0's pick-3 + trust-the-snippets
    strategy. The retriever can compare chunks across many sources rather
    than relying on the picker's snippet-based judgment.

    Phase 4: filter out URLs we've already fetched in earlier loop
    iterations BEFORE sending the candidate list to the model. Returning
    `picked_urls=[]` is the signal that we have nothing new to fetch — the
    conditional edge after sufficiency_check uses this as the
    "diminishing returns" exit, breaking the loop instead of burning
    another iteration that would surface the same pages.
    """
    fetched: set[str] = set(state.get("fetched_urls") or [])
    all_results = state.get("search_results") or []
    candidates = [r for r in all_results if r.url and r.url not in fetched]

    if not candidates:
        _status("[pick] 0 URLs — every candidate already fetched (diminishing returns)")
        _event("stage_data", stage="pick_links", picked=[], n_candidates=0,
               note="every candidate already fetched (diminishing returns)")
        return {"picked_urls": []}

    n_picks = min(N_LINKS, len(candidates))
    llm = get_llm()
    structured_llm = with_schema(llm, LinkPicks)
    messages = build_link_picker_messages(
        state["question"], candidates, n_picks=n_picks
    )
    picks: LinkPicks = await structured_llm.ainvoke(messages)
    # The picker LLM's structured output isn't guaranteed unique -- Gemma
    # 4 E2B has emitted the same URL twice in `picks.picks` before, which
    # then propagated into a duplicated fetch and a duplicated (mislabeled
    # under the fix in fetch.py, wasted otherwise) entry in `pages`.
    seen: set[str] = set()
    chosen: list[str] = []
    for p in picks.picks:
        if p.url in fetched or p.url in seen:
            continue
        seen.add(p.url)
        chosen.append(p.url)
    _status(f"[pick] {len(chosen)} URLs:")
    for u in chosen:
        _status(f"        {u}")
    _event("stage_data", stage="pick_links", picked=list(chosen),
           n_candidates=len(candidates))
    return {"picked_urls": chosen}


async def fetch_node(state: AnswerState) -> dict:
    """Fetch the picked URLs, returning markdown.

    v1 change: we no longer pass the search query as a BM25 filter. The
    per-page filter pre-cut content the retriever should be deciding on,
    and it can't compare across sources. Now retrieval is the only filter.
    The fallback PruningContentFilter still removes nav/chrome boilerplate.

    Phase 4: accumulate `pages` and `fetched_urls` across loop iterations.
    Each iteration's NEW pages are merged into the accumulated dict so
    that downstream checks (eval `cited_urls_in_fetched_pages`, trace
    inspection) see the full set of pages this run has ever pulled. The
    `picked_urls` list still represents only this iteration's picks, so
    `chunk_and_index_node` knows which subset is fresh.
    """
    picked = state.get("picked_urls") or []
    if not picked:
        _status("[fetch] skipped — nothing to fetch this iteration")
        _event("stage_data", stage="fetch", pages=[], n_ok=0, n_picked=0,
               note="nothing to fetch this iteration")
        return {}

    _status(f"[fetch] fetching {len(picked)} pages...")
    new_pages = await fetch_pages(picked)  # no query — drop BM25
    new_pages = {
        url: text for url, text in new_pages.items() if text and text.strip()
    }
    total_chars = sum(len(t) for t in new_pages.values())
    _status(
        f"[fetch] {len(new_pages)}/{len(picked)} pages OK "
        f"({total_chars // 1000}k chars)"
    )
    _event(
        "stage_data", stage="fetch", n_ok=len(new_pages), n_picked=len(picked),
        total_chars=total_chars,
        pages=[{"url": u, "chars": len(t)} for u, t in new_pages.items()],
    )

    accumulated_pages = {**(state.get("pages") or {}), **new_pages}
    accumulated_fetched = list(
        dict.fromkeys((state.get("fetched_urls") or []) + list(new_pages.keys()))
    )
    return {
        "pages": accumulated_pages,
        "fetched_urls": accumulated_fetched,
    }


async def chunk_and_index_node(state: AnswerState) -> dict:
    """Chunk this iteration's NEW pages, embed, persist in Chroma.

    Phase 4: only chunk pages corresponding to THIS iteration's
    `picked_urls`. The accumulated `pages` dict carries everything fetched
    so far, but on iteration 2 we don't want to re-chunk iteration 1's
    pages (Chroma's content-addressable upsert would dedup anyway, but
    re-chunking is wasted CPU and obscures the trace).

    Returns the freshly-produced Chunk list in `new_chunks` for trace
    visibility (so iteration 2's `new_chunks` shows the iter-2 deltas).
    Chunks already in the collection by ID collision are NOT re-embedded —
    that's the persistence win.
    """
    picked = state.get("picked_urls") or []
    all_pages = state.get("pages") or {}
    new_pages = {u: all_pages[u] for u in picked if u in all_pages}

    chunks = chunk_pages(new_pages)
    if not chunks:
        _status("[chunks] skipped — no new pages")
        _event("stage_data", stage="chunk_and_index", n_chunks=0,
               note="no new pages to chunk")
        return {"new_chunks": []}

    collection = get_collection(state["job_id"])
    n_new = upsert_chunks(collection, chunks)
    reused = len(chunks) - n_new
    _status(
        f"[chunks] {len(chunks)} this iter (+{n_new} new embeds"
        f"{f', {reused} reused' if reused else ''})"
    )
    _event("stage_data", stage="chunk_and_index", n_chunks=len(chunks),
           n_new=n_new, n_reused=reused)
    return {"new_chunks": chunks}


async def retrieve_node(state: AnswerState) -> dict:
    """Embed the query, pull top-K candidates from the job's Chroma collection.

    K=40 (config.TOP_K_RETRIEVAL): a wider candidate pool than we'll feed
    to the answer node, because the embedder's cosine ranking is coarse —
    the right chunk sometimes sits in position 25-35. The reranker (next
    node) sharpens the ordering, then MMR picks the final top-N=8.
    """
    collection = get_collection(state["job_id"])
    q_text = state["question"]  # see config.EMBED_QUERY_SOURCE — fixed for now
    q_emb = embed_query(q_text)
    hits = similarity_search(collection, q_emb, k=TOP_K_RETRIEVAL)
    top = hits[0].retrieval_score if hits else 0.0
    _status(f"[retrieve] {len(hits)} candidates (top cosine {top:.3f})")
    _event("stage_data", stage="retrieve", n_candidates=len(hits),
           top_cosine=round(float(top), 3))
    return {"retrieved": hits}


async def rerank_node(state: AnswerState) -> dict:
    """Cross-encoder rerank over the K=40 retrieval candidates.

    Each hit's `rerank_score` is populated and the list is sorted by
    rerank_score descending. The full list stays in state — MMR (next
    node) needs all K to compute diversity against the relevance ranking.

    Cross-encoder relevance is a much sharper signal than dense cosine;
    candidates that were tied around 0.55 cosine often spread out across
    a wide rerank range, which is exactly what we want for downstream
    picking.
    """
    hits = state.get("retrieved", []) or []
    if not hits:
        _status("[rerank] skipped — no candidates")
        _event("stage_data", stage="rerank", n=0, note="no candidates")
        return {"retrieved": []}
    reranked = rerank(state["question"], hits)
    hi = reranked[0].rerank_score or 0.0
    lo = reranked[-1].rerank_score or 0.0
    _status(f"[rerank] scored {len(reranked)} candidates, range [{lo:+.2f} .. {hi:+.2f}]")
    _event("stage_data", stage="rerank", n=len(reranked),
           lo=round(float(lo), 2), hi=round(float(hi), 2))
    return {"retrieved": reranked}


async def select_chunks_node(state: AnswerState) -> dict:
    """MMR pick top-N from the reranked candidates, then group by source.

    MMR uses `rerank_score` for relevance and chunk embeddings for
    diversity (cosine). This is the architectural fix for the Phase 1
    failure mode where 6/8 chunks came from one URL — MMR penalizes
    candidates that are too similar to already-picked ones, which
    naturally drives source variety when sources are topically near.

    After MMR, we reorder the N picks so chunks from the same URL sit
    together (best-scoring URL's chunks first). The answer model writes
    more coherent prose when it can discuss one source at a time rather
    than zig-zagging between domains.
    """
    candidates = state.get("retrieved", []) or []
    selected = select_mmr(candidates, lambda_=MMR_LAMBDA, top_n=TOP_N_ANSWER)
    selected = _group_chunks_for_prompt(selected)
    domains = sorted({h.domain for h in selected})
    selected_ids = {h.chunk_id for h in selected}
    _status(
        f"[select] {len(selected)} chunks across {len(domains)} "
        f"domain{'s' if len(domains) != 1 else ''}: {domains}"
    )
    _event(
        "stage_data", stage="select_chunks", domains=domains,
        chunks=[{"id": h.chunk_id, "domain": h.domain, "url": h.url,
                 "rerank": round(float(h.rerank_score), 2)
                 if h.rerank_score is not None else None,
                 "retrieval": round(float(h.retrieval_score), 3)
                 if h.retrieval_score is not None else None,
                 "text": (h.text[:240] + "…") if len(h.text) > 240 else h.text}
                for h in selected],
        # the FULL reranked pool (top-K) with a kept/dropped flag — drives the
        # 40 → 12 funnel view in retrieve/rerank, same as the frozen run.json.
        candidates=[{"id": h.chunk_id, "domain": h.domain, "url": h.url,
                     "retrieval": round(float(h.retrieval_score), 4)
                     if h.retrieval_score is not None else None,
                     "rerank": round(float(h.rerank_score), 4)
                     if h.rerank_score is not None else None,
                     "selected": h.chunk_id in selected_ids}
                    for h in candidates],
    )
    return {"selected": selected}


async def sufficiency_node(state: AnswerState) -> dict:
    """Decide whether the selected chunks are good enough to answer over.

    LLM judge that READS the chunks (see `sufficiency.py` — the v3 verdict
    prompt). The result is stashed in state for the conditional edge to read.
    We also bump `iteration` here — the natural "one loop iteration just
    completed" point.

    The conditional edge after this node (`route_after_sufficiency`)
    inspects three signals:
        1. `sufficiency.sufficient`   — did this iteration find good evidence?
        2. `iteration >= max_iterations` — have we used our budget?
        3. `picked_urls == []`        — diminishing returns (no new URLs to fetch)?
    Any of (2) or (3) sends us to `answer` even on insufficient evidence;
    the prompt's "if sources are insufficient, disclaim" rule handles the
    weak-answer path.
    """
    selected = state.get("selected") or []
    result = await judge_sufficiency(state["question"], selected)
    iteration_out = state.get("iteration", 0) + 1
    mark = "✓ sufficient" if result.sufficient else "✗ insufficient"
    _status(
        f"[sufficiency] iter {iteration_out}: {mark} "
        f"({result.n_chunks} chunks)"
    )
    _status(f"             reason: {result.reason}")
    _event(
        "stage_data", stage="sufficiency_check", iteration=iteration_out,
        sufficient=bool(result.sufficient), reason=result.reason,
        n_chunks=result.n_chunks,
    )
    return {"sufficiency": result, "iteration": iteration_out}


def route_after_generate(state: AnswerState) -> str:
    """Conditional-edge router from `generate_queries`.

    Normally proceeds to `search`. But when a refinement pass set `refine_stop`
    (the gap is unfillable by search, or no usable new query was produced),
    skip the doomed search/fetch/rerank cycle and go straight to `answer` — the
    prior iteration's insufficient `sufficiency` verdict is still in state, so
    the answer node takes the disclaim path over what we already have.
    """
    if state.get("refine_stop"):
        return "answer"
    return "search"


def route_after_sufficiency(state: AnswerState) -> str:
    """Conditional-edge router from `sufficiency_check`.

    Returns the name of the next node:
    - `"answer"` when we should commit (sufficient, or out of budget, or
      diminishing returns).
    - `"generate_queries"` when we should loop back for a refinement pass.

    Order matters: we check `sufficient` first so a happy-path question
    doesn't waste a check; then the budget guard (`iteration`); then the
    diminishing-returns guard (no new URLs picked).
    """
    suf = state.get("sufficiency")
    if suf and suf.sufficient:
        return "answer"

    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", MAX_ITERATIONS)
    if iteration >= max_iter:
        return "answer"

    # If this iteration's pick_links_node returned 0 picks (every search
    # result was already in fetched_urls), looping back would just produce
    # the same effect — no new URLs would be fetched and no new chunks
    # added to the collection. Exit cleanly.
    if not (state.get("picked_urls") or []):
        return "answer"

    return "generate_queries"


#: Directive appended when the sufficiency judge marked the sources
#: insufficient — primes the freeform draft to disclaim honestly rather than
#: manufacture an answer (the disclaim-001 win: freeform+thinking disclaims
#: correctly on its own; this just tells it WHY the sources fell short).
def _disclaim_directive(reason: str) -> str:
    return (
        "Note: an automated check judged these sources may NOT fully contain "
        f"the answer (reason: {reason}). If that is right, do not manufacture "
        "an answer — say plainly what the sources do not establish, then give "
        "whatever adjacent facts they DO contain."
    )


async def run_answer_stage(
    question: str,
    hits: list[RetrievalHit],
    sufficiency: SufficiencyResult | None,
    *,
    answer_prompt_builder: Callable[..., list] = build_answer_messages_freeform,
) -> dict:
    """The answer stage: draft + ground (the `answer` node). Citation is separate.

    Ditched the A path (structured `LLMAnswer` citation contract + holistic
    critique) after the A-vs-B read: A under-answered multi-facet questions and
    fabricated on the disclaim cases, both of which the freeform+thinking draft
    fixes. This function owns the two COUPLED steps:

    1. **Draft** — freeform + thinking (`build_answer_messages_freeform`,
       `reasoning=True`). No citation apparatus, so synthesis is unconstrained;
       on an insufficient verdict a disclaim directive primes an honest
       disclaim instead of a manufactured answer.
    2. **Grounding gate** — the attribution pass maps every claim to its
       chunk(s); any `supported=false` claim (a hallucination) triggers a
       bounded re-draft with the flagged claims fed back, up to
       `MAX_CRITIQUE_ATTEMPTS` attempts.

    Step 3 (**cite**) is NOT here — it is `cite_draft`, its own deterministic,
    model-free node. Draft and grounding are coupled by a feedback loop so they
    stay together; citation is a pure terminal add-on, so it is its own stage
    (node-per-stage principle). This returns the grounded
    freeform draft; the `cite` node turns it into the `[n]`-annotated answer.

    No separate disclaim path and no holistic critique: the freeform draft
    self-disclaims, attribution owns grounding, and sufficiency/refinement
    already gated "are these sources any good" upstream. Termination is bounded
    (≤ `MAX_CRITIQUE_ATTEMPTS` draft calls, each with an attribution call).

    `answer_prompt_builder` stays injectable so the eval harness can swap a
    draft-prompt variant without editing this function.

    Returns `{draft, attribution?, answer_attempts?}` — NOT the cited answer.
    """
    insufficient = sufficiency is not None and not sufficiency.sufficient
    directive = (
        _disclaim_directive(sufficiency.reason)
        if insufficient and sufficiency is not None
        else None
    )

    draft_llm = get_llm(reasoning=True)  # freeform draft: thinking on
    attrib_structured = with_schema(get_llm(), AttributionResult).with_retry(
        stop_after_attempt=2,
    )

    feedback: str | None = None
    draft = ""
    attribution: AttributionResult | None = None
    for attempt in range(1, MAX_CRITIQUE_ATTEMPTS + 1):
        label = "regenerating" if feedback else "drafting"
        _status(
            f"[answer] {label} (freeform+thinking) from {len(hits)} chunks "
            f"(attempt {attempt}/{MAX_CRITIQUE_ATTEMPTS})..."
        )
        messages = build_answer_messages_freeform(
            question, hits,
            critique_feedback=feedback,
            answer_directives=directive,
        )
        resp = await draft_llm.with_retry(stop_after_attempt=2).ainvoke(messages)
        draft = (resp.content or "").strip()

        _status(f"[answer] grounding review (attempt {attempt})...")
        attribution = await attrib_structured.ainvoke(
            build_attribution_messages(question, draft, hits)
        )
        unsupported = [c.claim for c in attribution.claims if not c.supported]
        if not unsupported:
            _status(f"[answer] ✓ all {len(attribution.claims)} claims grounded")
            break
        _status(f"[answer] ✗ {len(unsupported)} unsupported claim(s) — re-draft")
        feedback = (
            "Claims NOT supported by the sources (remove or correct — do not "
            "restate them):\n" + "\n".join(f"- {u}" for u in unsupported)
        )

    # Citation is NOT done here — it is the `cite` node, a separate, deterministic,
    # model-free stage (see `cite_draft` below). This node returns the grounded freeform draft and its
    # attribution; `cite` turns that into the final `[n]`-annotated answer. On a
    # re-draft only the terminal draft reaches here, so `draft` is that draft.
    return {
        "draft": draft,
        "attribution": attribution,
        "answer_attempts": attempt,
    }


def cite_draft(
    draft: str,
    attribution: AttributionResult | None,
    hits: list[RetrievalHit],
) -> Answer:
    """The CITE stage (draft-then-cite, step 3): deterministic, model-free.

    Its own function — and its own graph node (`cite_node`) — because it is a
    pure, terminal transform with NO feedback loop: it annotates the grounded
    freeform `draft` in place with `[n]` markers (`cite.annotate_in_place`,
    containment matching, no model) and returns the final `Answer` (cited text +
    flattened per-URL citations + the claims kept for audit). Citation is
    assigned AFTER generation, never during it.

    Splitting it out is the node-per-stage principle applied where it is cleanest:
    the draft/attribute/regenerate loop stays coupled in `run_answer_stage`, and
    this pure add-on becomes independently inspectable and freezable
    (`eval/freeze_annotations.py`).
    """
    text, citations, placements = (
        annotate_in_place(draft, attribution, hits)
        if attribution is not None else (draft, [], [])
    )
    return Answer(
        text=text,
        citations=citations,
        placements=placements,
        claims=attribution.claims if attribution else [],
    )


async def answer_node(state: AnswerState) -> dict:
    """Graph node for the answer stage: draft + ground (NOT cite).

    Thin by design: it pulls the frozen inputs (selected chunks, sufficiency
    verdict, question) out of graph state and delegates to `run_answer_stage`,
    which owns the coupled draft → attribute → regenerate loop. Keeping the logic
    in a standalone function is what lets the eval replay harness exercise the
    exact same stage offline against trace-frozen chunks.

    Emits the grounded `draft`, its `attribution`, and the attempt count. Citation
    is the next node (`cite_node`); word/citation counts are logged there.
    """
    out = await run_answer_stage(
        state["question"],
        state.get("selected", []) or [],
        state.get("sufficiency"),
    )
    attrib = out.get("attribution")
    n_unsupported = (
        sum(1 for c in attrib.claims if not c.supported) if attrib else None
    )
    _event(
        "stage_data", stage="answer",
        draft_words=len((out.get("draft") or "").split()),
        answer_attempts=out.get("answer_attempts"),
        unsupported_claims=n_unsupported,
    )
    return out


async def cite_node(state: AnswerState) -> dict:
    """Graph node for the CITE stage: annotate the grounded draft in place.

    Deterministic and model-free — reads the `draft` + `attribution` the answer
    node produced (and the `selected` chunks for `chunk_id` → URL) and delegates
    to `cite_draft`, which returns the final `[n]`-annotated `Answer`. Its own
    node because it is a pure terminal transform with no feedback loop; that is
    also what makes it independently freezable (`eval/freeze_annotations.py`).
    """
    answer = cite_draft(
        state.get("draft", "") or "",
        state.get("attribution"),
        state.get("selected", []) or [],
    )
    _event(
        "stage_data", stage="cite",
        n_citations=len(answer.citations),
        words=len(answer.text.split()),
    )
    return {"final_answer": answer}


# --- Graph -------------------------------------------------------------------


def build_graph():
    g = StateGraph(AnswerState)
    # Each node is wrapped in `_instrument` so the structured event channel gets
    # stage_start / stage_end (with elapsed timing) for free — the UI's per-stage
    # timers come from these, no per-node timing code required.
    g.add_node("generate_queries", _instrument("generate_queries", generate_queries_node))
    g.add_node("search", _instrument("search", search_node))
    g.add_node("pick_links", _instrument("pick_links", pick_links_node))
    g.add_node("fetch", _instrument("fetch", fetch_node))
    g.add_node("chunk_and_index", _instrument("chunk_and_index", chunk_and_index_node))
    g.add_node("retrieve", _instrument("retrieve", retrieve_node))
    g.add_node("rerank", _instrument("rerank", rerank_node))
    g.add_node("select_chunks", _instrument("select_chunks", select_chunks_node))
    g.add_node("sufficiency_check", _instrument("sufficiency_check", sufficiency_node))
    g.add_node("answer", _instrument("answer", answer_node))
    g.add_node("cite", _instrument("cite", cite_node))

    g.add_edge(START, "generate_queries")
    # Normally generate_queries -> search; a refinement stop routes to answer.
    g.add_conditional_edges(
        "generate_queries",
        route_after_generate,
        {
            "search": "search",
            "answer": "answer",
        },
    )
    g.add_edge("search", "pick_links")
    g.add_edge("pick_links", "fetch")
    g.add_edge("fetch", "chunk_and_index")
    g.add_edge("chunk_and_index", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "select_chunks")
    g.add_edge("select_chunks", "sufficiency_check")
    # Phase 4 loop: conditional edge from sufficiency_check either loops
    # back to generate_queries (refinement) or proceeds to answer.
    g.add_conditional_edges(
        "sufficiency_check",
        route_after_sufficiency,
        {
            "generate_queries": "generate_queries",
            "answer": "answer",
        },
    )
    # Answer stage is two nodes: draft+ground (`answer`) then the deterministic,
    # model-free citation (`cite`). Both the normal path and the disclaim path
    # (refinement STOP / MAX_ITERATIONS routing straight to `answer`) flow through
    # `cite` before END.
    g.add_edge("answer", "cite")
    g.add_edge("cite", END)

    return g.compile()


# Compiled once at module load — same instance reused across runs.
GRAPH = build_graph()


async def answer_question(question: str, *, job_id: str | None = None) -> AnswerState:
    """Run the graph end-to-end and return the final state.

    `job_id` namespaces the Chroma collection. Pass an explicit id to
    reuse / accumulate a corpus across multiple questions in one research
    session; leave None for one-off ad-hoc questions and a fresh uuid is
    minted per call (eval-friendly default — no cross-question leakage).

    The loop state is seeded here (iteration=0, max_iterations from
    config). LangGraph has a default recursion limit of 25 super-steps;
    our worst-case loop is well under that (2 iterations × 9 nodes =
    18), so we don't need to override it.
    """
    jid = job_id or f"adhoc_{uuid.uuid4().hex[:12]}"
    _status(f"\n[Q] {question}")
    _status(f"[job] {jid}")
    initial: AnswerState = {
        "question": question,
        "job_id": jid,
        "iteration": 0,
        "max_iterations": MAX_ITERATIONS,
        "query_history": [],
        "fetched_urls": [],
        "pages": {},
    }
    return await GRAPH.ainvoke(initial)
