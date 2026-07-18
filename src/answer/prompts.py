"""Prompt builders and structured-output schemas for the Answer pipeline.

Conventions (see memory: prompt_conventions.md, pydantic_structured_output.md):
- Every system prompt starts with `_datetime_context_block(now)`.
- Structured-output prompts inject their schema via `pydantic_to_prompt(Model)`,
  paired with at least one example output.
- Schemas live next to the prompt builder that uses them.
"""
from __future__ import annotations

import typing
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from .search import SearchResult


# --- Shared helpers ----------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _datetime_context_block(now: datetime | None = None) -> str:
    """Standard date/time stamp prepended to every system prompt.

    UTC, ISO-8601 with `Z`, plus weekday for natural-language readability.
    """
    n = now or _now_utc()
    return (
        f"Current date and time (UTC): {n.strftime('%Y-%m-%d %H:%M:%SZ')} "
        f"({n.strftime('%A')})"
    )


def _render_type(tp) -> tuple[str, type[BaseModel] | None]:
    """Render a Python/Pydantic type as a human-readable label.

    Returns (label, nested_model). The nested_model is non-None when the type
    is (or contains) a BaseModel subclass we should recurse into.
    """
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    # Optional[X] / X | None
    if origin in (typing.Union, getattr(typing, "UnionType", typing.Union)):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            label, nested = _render_type(non_none[0])
            return f"{label} or null", nested

    # list[X]
    if origin in (list, typing.List):  # noqa: UP006
        if args:
            inner_label, nested = _render_type(args[0])
            return f"list of {inner_label}", nested
        return "list", None

    # dict[K, V]
    if origin in (dict, typing.Dict):  # noqa: UP006
        return "object (dict)", None

    # Pydantic BaseModel subclass
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return f"object ({tp.__name__})", tp

    # Primitives
    if tp is str:
        return "string", None
    if tp is int:
        return "integer", None
    if tp is float:
        return "number", None
    if tp is bool:
        return "boolean", None

    return getattr(tp, "__name__", str(tp)), None


def pydantic_to_prompt(model: type[BaseModel], indent: int = 0) -> str:
    """Render a Pydantic v2 model as a human-readable schema block for prompts.

    Walks `model.model_fields`, formatting each field as a markdown bullet with
    its type and description. Recurses into nested BaseModel types. Skips JSON
    Schema bloat (`$defs`, `anyOf`, enums) — what reaches the LLM is just the
    field semantics. The bullet form is what small models actually read.
    """
    pad = "  " * indent
    lines: list[str] = []
    for name, info in model.model_fields.items():
        type_label, nested = _render_type(info.annotation)
        desc = (info.description or "").strip()
        line = f"{pad}- {name} ({type_label})"
        if desc:
            line += f": {desc}"
        lines.append(line)
        if nested is not None:
            lines.append(f"{pad}  Each {nested.__name__} has these fields:")
            lines.append(pydantic_to_prompt(nested, indent=indent + 2))
    return "\n".join(lines)


# --- Stage 1: query planning (structured) -----------------------------------


class GeneratedQueries(BaseModel):
    """1 to 3 search queries that together cover the user's question.

    The model decides the count. The schema enforces only the hard bounds
    (min 1, max 3) so a confused planner can't fan out to 10 and burn the
    search budget; within those bounds the model is free to pick 1, 2, or
    3 based on its reading of the question.

    `reasoning` is captured so the trace shows *why* the planner picked
    this count — invaluable when iterating the planning prompt later.
    """

    reasoning: str = Field(
        description=(
            "One short sentence (≤30 words) stating the query-count "
            "decision in this EXACT format: \"I will emit N queries "
            "because <reason>\". Name N (1, 2, or 3) explicitly at the "
            "start. This commits you to the count BEFORE generating the "
            "queries list — the number of items in `queries` MUST match "
            "the N you state here. Example: \"I will emit 1 query "
            "because this is a single definitional intent.\""
        )
    )
    queries: list[str] = Field(
        min_length=1,
        max_length=3,
        description=(
            "Between 1 and 3 web search queries that together cover the "
            "question. Each query: short (typically 3-8 keywords); drops "
            "filler words like 'what is', 'tell me about'; preserves named "
            "entities, technical terms, and product names verbatim. Do NOT "
            "include quotes, labels, or 'Query:' prefixes — emit each query "
            "as a bare keyword string. Don't repeat the same query in "
            "different phrasings."
        ),
    )


_QUERY_PLANNER_EXAMPLES_TEMPLATE = """Question: who founded huggingface and when?
Context: first attempt, no prior evidence.
Decision: One query — single factoid intent, one search will surface this.
Output:
{{"reasoning": "I will emit 1 query because this is a single factoid about one entity.",
  "queries": ["huggingface founders history"]}}

Question: what is retrieval-augmented generation?
Context: first attempt, no prior evidence.
Decision: One query — single definitional intent.
Output:
{{"reasoning": "I will emit 1 query because this is a single definitional intent.",
  "queries": ["retrieval augmented generation rag definition"]}}

Question: how does langgraph differ from langchain?
Context: first attempt, no prior evidence.
Decision: Two queries — a comparison of two distinct entities, each benefits from its own search.
Output:
{{"reasoning": "I will emit 2 queries because this is a comparison; one targeted query per side.",
  "queries": ["langgraph features architecture", "langchain features architecture"]}}

Question: what is langgraph, who built it, and what are its main use cases?
Context: first attempt, no prior evidence.
Decision: Three queries — three genuinely distinct facets.
Output:
{{"reasoning": "I will emit 3 queries because this has three distinct facets: definition, origin, use cases.",
  "queries": ["langgraph framework definition", "langchain langgraph creator", "langgraph use cases applications"]}}

Question: what was the topic of the keynote at TechConf 2026?
Context: follow-up iteration. Prior queries tried: ["techconf 2026 keynote topic"]. Prior evidence found pages about TechConf's general programs and sponsors, but nothing about a specific keynote or the event's published schedule.
Decision: Pivot — search the event itself (program/schedule) rather than its content. One focused query.
Output:
{{"reasoning": "I will emit 1 query because the prior search missed the event program; pivot to the schedule/announcement angle.",
  "queries": ["techconf 2026 program schedule speakers announcement"]}}"""


def _format_query_history(history: list[str]) -> str:
    """Render the 'what's been tried' block of the planner's user message.

    On the first iteration, no queries have been tried yet — we say so
    explicitly rather than emitting an empty list, so the model never
    has to interpret a missing block.
    """
    if not history:
        return "(This is the first attempt — no prior queries yet.)"
    return "\n".join(f"- {q}" for q in history)


def _summarize_prior_evidence(
    hits: list,  # list[RetrievalHit] — duck-typed for .text / .domain
    *,
    max_per_chunk: int = 300,
    max_chunks: int = 3,
) -> str:
    """Render the 'what's been found' block of the planner's user message.

    Shows the top N retrieved chunks with their source domain and a
    truncated text snippet so the planner can see what the prior search
    actually surfaced. Without this, the planner has no grounded sense
    of where the prior queries landed and ends up repeating the same
    intent in different words (the failure mode we saw on iteration 2).

    `max_chunks=3` keeps the planner prompt compact; the marginal chunks
    are usually noise anyway. `max_per_chunk=300` is enough to make the
    topic recognizable without bloating the prompt.
    """
    if not hits:
        return "(No evidence retrieved yet — this is the first search.)"
    parts: list[str] = []
    for h in hits[:max_chunks]:
        text = " ".join(h.text.split())  # collapse internal whitespace
        if len(text) > max_per_chunk:
            text = text[:max_per_chunk].rstrip() + "..."
        parts.append(f"- [{h.domain}] {text}")
    return "\n".join(parts)


def build_query_generation_messages(
    question: str,
    *,
    query_history: list[str] | None = None,
    prior_evidence: list | None = None,  # list[RetrievalHit]
    sufficiency_reason: str | None = None,
    now: datetime | None = None,
) -> list:
    """System+user messages for planning 1-3 search queries.

    Iteration-agnostic. The same function handles both initial planning
    (no prior context) and refinement planning (with prior queries +
    retrieved evidence + the sufficiency verdict). The user message's
    context blocks fill themselves in based on what's passed — on iter 1
    the placeholder lines say "first attempt, no prior queries yet" and
    the model plans accordingly; on iter 2+ the planner sees the goal,
    what's been tried, what's been found, and why it was insufficient,
    and can reason about the next angular shift.

    Structured output: returned message validates against `GeneratedQueries`.
    The schema's `max_length=3` is a HARD cap — if the model emits 4+
    queries, pydantic raises and the call fails loud (no silent truncation,
    no retry). Use this signal to iterate the prompt rather than masking
    over-emission.

    The decision-criteria block in the system prompt is the calibrator
    that keeps small models from defaulting to "always emit the max" —
    they need concrete instructions on when one query is enough.
    """
    n = now or _now_utc()
    year = n.year
    schema_block = pydantic_to_prompt(GeneratedQueries)

    system = f"""{_datetime_context_block(n)}

You are a search-query planner. Given a question and the context of what (if anything) has already been searched, plan 1 to 3 web search queries optimized for a metasearch engine (Google, Bing, DuckDuckGo) that will surface the information needed to answer it.

How to decide the number of queries — choose deliberately:

- Use **one** query when the question has a single focused intent that a good search will satisfy in one shot (factoid, definitional, simple how-to questions).
- Use **two** queries when the question has two clearly distinct sub-topics that one search couldn't cover well together (e.g. comparisons between two entities).
- Use **three** queries only when the question has three genuinely distinct facets, each of which would benefit from its own targeted search.

Pick the smallest number that actually covers the question. Don't fan out for the sake of fanning out, and don't under-search a multi-faceted question. Be just where you need to be — no greedier, no stingier.

For each query:
- Short — typically 3 to 8 keywords.
- Drop filler ("what is", "tell me about", "the", "a").
- Preserve named entities, technical terms, and product names verbatim.
- For time-sensitive questions about the current state of the world, append "{year}" to favor recent results.

If this is a follow-up iteration — the user message will list prior queries and the evidence they found — you MUST take a different angle:
- Do NOT repeat any of the queries in the "what's been tried so far" list, even with minor rewording.
- Use the evidence shown to reason about what the prior search missed. What aspect of the question is not covered? Would searching for the event/entity/program itself surface more (rather than its content)? Are there alternative names, dates, related organizations, or different terminology to try? Sometimes broader is better; sometimes narrower is better.
- The same decision rule for 1, 2, or 3 queries applies — just make them substantively different from what was already tried.

Output schema (JSON):

{schema_block}

Examples:

{_QUERY_PLANNER_EXAMPLES_TEMPLATE}
"""

    history_block = _format_query_history(query_history or [])
    evidence_block = _summarize_prior_evidence(prior_evidence or [])
    reason_block = (
        f"\n\nWhy the prior evidence was judged insufficient:\n{sufficiency_reason}"
        if sufficiency_reason
        else ""
    )

    user = f"""Goal — answer this question: {question}

What's been tried so far:
{history_block}

What's been found so far:
{evidence_block}{reason_block}"""

    return [SystemMessage(content=system), HumanMessage(content=user)]


# --- Stage 1 (simple variant): natural "search like a person" planner --------
# An experiment against `build_query_generation_messages`: instead of a
# rulebook (count-decision criteria, keyword-length rules, facet lecture),
# just ask the model what it would type into a search engine and to justify
# each query. Structured output, per-query justification, and NO count cap —
# the model uses as many queries as it genuinely wants. Kept as a separate
# schema + builder so it can be A/B'd against the baseline planner via the
# query playground without disturbing the pipeline's planner.


class PlannedQuery(BaseModel):
    """One search query plus the reason you'd run it."""

    query: str = Field(
        description=(
            "A search query you would type into a search engine to help "
            "answer the question. Bare keywords — no quotes, no 'Query:' "
            "prefix, no numbering."
        )
    )
    why: str = Field(
        description=(
            "One short sentence: why you'd search this, and how it helps "
            "answer the question."
        )
    )


class PlannedQueries(BaseModel):
    """The search queries you'd run to answer the question, each justified.

    Deliberately has NO min/max cap on the number of queries (beyond
    'at least one') — the point of this variant is to see how many the
    model reaches for when it isn't told a limit.
    """

    queries: list[PlannedQuery] = Field(
        min_length=1,
        description=(
            "The search queries you would run to answer the question, each "
            "with its justification. Use as many or as few as you genuinely "
            "need — no more, no fewer."
        ),
    )


class RefinedQueries(BaseModel):
    """Refinement-pass output: NEW queries, OR an explicit stop signal.

    The refinement planner (loop iteration > 0) differs from the first-pass
    `PlannedQueries` in one structural way: it can decide the missing
    information is **not findable by any search** and stop, instead of being
    forced to emit at least one query. `PlannedQueries.queries` has
    `min_length=1`, so on an unanswerable question (Anthropic's private Q3
    revenue, an appointment that hasn't happened) the loop would otherwise
    churn out ≥1 new query every iteration until the budget runs out. This
    schema gives it a way out.

    `reason` precedes `searchable` (commit after justifying — same ordering as
    the sufficiency judge). `searchable` is POSITIVELY named (true = a search
    can plausibly find it = the continue path), so the plain-English reading
    and the code meaning agree — avoiding the polarity trap that inverted a
    problem-named bool on E4B.
    """

    reason: str = Field(
        description="Brief: what was missing, and whether a further web search "
                    "could plausibly find it."
    )
    searchable: bool = Field(
        description="TRUE if a further internet search could plausibly find the "
                    "missing information — then give new queries below. FALSE if "
                    "it likely cannot be found by any search (not public, may not "
                    "exist yet) — then give NO queries; the loop stops and "
                    "disclaims."
    )
    queries: list[PlannedQuery] = Field(
        default_factory=list,
        description="New search queries targeting the gap, each genuinely "
                    "different from what was already tried. EMPTY when "
                    "searchable is false.",
    )


_SIMPLE_QUERY_EXAMPLE = """{
  "queries": [
    {"query": "langgraph framework what is",
     "why": "Pins down what LangGraph actually is, which any answer has to define first."},
    {"query": "langgraph use cases production examples",
     "why": "Finds where it's used in practice, covering the 'what it's for' part of the question."}
  ]
}"""


def build_query_generation_messages_simple(
    question: str,
    *,
    extra_guidance: str = "",
    now: datetime | None = None,
) -> list:
    """Minimal, natural-framing planner prompt (experimental variant).

    No count-decision rules, no keyword-length rules, no facet lecture —
    just: "here's a question, what would you search for, and why?" The
    structured output (`PlannedQueries`) captures one justification per
    query. First-attempt only (no refinement-iteration handling); this
    variant exists to probe whether a simpler prompt yields better, less
    redundant queries than the rulebook baseline.

    `extra_guidance` is an optional block spliced in after the framing —
    the seam the query playground uses to A/B targeted tweaks (intent→signal
    translation, relative-date resolution, a soft count cap) against the
    bare `simple` prompt without forking a whole new builder each time.
    """
    n = now or _now_utc()
    schema_block = pydantic_to_prompt(PlannedQueries)
    guidance = f"{extra_guidance.strip()}\n\n" if extra_guidance.strip() else ""

    system = f"""{_datetime_context_block(n)}

Someone asked you the question below and wants you to find the answer using an internet search engine. Write the search query — or queries — you would actually type in to find that answer. For each one, briefly say why you'd search it and how it helps answer the question.

Search the way a thoughtful person would: clear, effective queries. Use as many or as few as you genuinely need — no more, no fewer.

{guidance}Output schema (JSON):

{schema_block}

Example output:
{_SIMPLE_QUERY_EXAMPLE}
"""

    user = f"Question: {question}"
    return [SystemMessage(content=system), HumanMessage(content=user)]


_NEEDS_QUERY_EXAMPLES = """Question: How much did Anthropic raise in its latest funding round, and who led it?
Output:
{
  "queries": [
    {"query": "Anthropic latest funding round amount raised",
     "why": "Finds the size of the round — the first piece the question needs."},
    {"query": "Anthropic funding round lead investor",
     "why": "Finds who led it — a separate piece, since the amount and the lead are reported as distinct facts."}
  ]
}

Question: Who did OpenAI appoint as its head of safety last week?
Output:
{
  "queries": [
    {"query": "OpenAI new head of safety appointment",
     "why": "Finds the appointment. 'last week' is dropped — a search engine can't use a relative date, and recent coverage surfaces it anyway."}
  ]
}

Question: Compare Pinecone, Weaviate, Qdrant, and Chroma on performance and pricing.
Output:
{
  "queries": [
    {"query": "Pinecone Weaviate Qdrant Chroma performance benchmark comparison",
     "why": "Covers the performance dimension for ALL FOUR at once — a single comparison search surfaces them together, so no query per store."},
    {"query": "Pinecone Weaviate Qdrant Chroma pricing comparison",
     "why": "Covers the pricing dimension for all four; even a four-way comparison decomposes by dimension (performance, pricing), never one query per item."}
  ]
}"""


def build_query_generation_messages_needs(
    question: str,
    *,
    prior_queries: list[str] | None = None,
    sufficiency_reason: str | None = None,
    now: datetime | None = None,
) -> list:
    """Information-needs planner: decompose by what you need to KNOW.

    Reframes the task around the distinct pieces of information required to
    answer the question — one query per piece, non-redundant, minimal — and
    explicitly instructs search-native rephrasing (drop the question's
    wording, filler, and relative dates). No in-prompt schema block (that
    proved redundant given `with_structured_output`); the worked examples do
    the heavy lifting (on this small model examples beat rules): per-piece
    decomposition, dropping a relative date, and decomposing a comparison by
    dimension rather than by item. Structured via
    `with_structured_output(PlannedQueries)`.

    **Refinement mode (loop-aware).** On a refinement pass the sufficiency
    judge has already run and found the first attempt's sources wanting. Pass
    `prior_queries` (everything tried so far) and `sufficiency_reason` (the
    judge's grounded account of what was missing) and a REFINEMENT block is
    appended, steering the planner to target the gap and NOT reissue the same
    queries. **First-pass rendering is byte-identical to before:** with both
    left as `None` (the default, `iteration == 0`) nothing is appended, so the
    validated first-attempt behavior is untouched.
    """
    n = now or _now_utc()

    system = f"""{_datetime_context_block(n)}

Someone asked you the question below and wants to find the answer using an internet search engine.

First, think about the specific pieces of information you would need in order to answer it. Then write one search query for each distinct piece. Do not be redundant: if two pieces would be found by the same search, use a single query. Fewer queries are better — as long as every piece the question needs is covered. When the question compares several things, decompose by the DIMENSION being compared (price, speed, architecture, …), not by each thing — a single query per dimension usually surfaces all of them at once, so you rarely need a query per item.

Write each query the way you would actually type it into a search engine, NOT by echoing the question's wording. Use the keywords and phrasing that surface good results: drop conversational filler, and drop relative time like "last week" or "recently" (they do not help a search engine). For each query, note which piece of information it is meant to find.

Examples:

{_NEEDS_QUERY_EXAMPLES}"""

    if prior_queries or sufficiency_reason:
        system += "\n\n" + _refinement_block(prior_queries, sufficiency_reason)

    user = f"Question: {question}"
    return [SystemMessage(content=system), HumanMessage(content=user)]


def _refinement_block(
    prior_queries: list[str] | None, sufficiency_reason: str | None
) -> str:
    """The REFINEMENT section appended on a loop iteration (>0).

    Two jobs, in priority order:
      1. **Don't repeat.** The prior queries are listed verbatim under an
         explicit do-not-reissue instruction — the single most important line,
         since without it the model re-plans the same searches and the loop
         spins in place.
      2. **Target the gap.** The judge's reason is handed over as "what was
         missing", with instructions to aim the new queries at THAT, via a
         different angle / more specific terms — not to re-decompose the
         original question.

    Kept example-free (per the no-worked-examples decision), and it carries an
    escape hatch: if the gap looks unfillable by search (the information may
    simply not be public — the genuine-disclaim case the judge now catches),
    returning FEWER queries is explicitly better than padding with
    near-duplicates. This is what stops an unanswerable question from thrashing
    the loop every iteration.
    """
    parts = ["## Refinement",
             "A previous search attempt was judged INSUFFICIENT — the sources "
             "it found did not answer the question."]

    if prior_queries:
        tried = "\n".join(f"- {q}" for q in prior_queries)
        parts.append(
            "Already tried — do NOT repeat these or lightly reword them:\n"
            f"{tried}")

    if sufficiency_reason:
        parts.append(f"What was missing:\n{sufficiency_reason}")

    parts.append(
        "First decide whether a further internet search could plausibly find "
        "what was missing. If it could, set searchable = true and plan NEW "
        "queries that go after the gap — a different angle, more specific or "
        "more authoritative terms — never a re-run or light rewording of what "
        "was already tried. If the missing information likely CANNOT be found "
        "by any search (it is not public, or may not exist yet), set "
        "searchable = false and return NO queries — do not pad with "
        "near-duplicates of the failed attempts; the loop will stop and "
        "disclaim honestly.")

    return "\n\n".join(parts)


def build_query_generation_messages_bare(
    question: str,
    *,
    now: datetime | None = None,
) -> list:
    """Simple planner with NO schema block and NO example in the prompt.

    Still meant to be called WITH `with_structured_output(PlannedQueries)`:
    the model receives the schema through the structured-output binding
    (Ollama's `format` field), just not a SECOND time as human-readable text
    in the prompt. This is the control for "is the in-prompt schema +
    example redundant given the bound schema?" — same framing as `simple`,
    minus the `pydantic_to_prompt` block and the worked example.
    """
    n = now or _now_utc()

    system = f"""{_datetime_context_block(n)}

Someone asked you the question below and wants to find the answer using an internet search engine. Write the search query — or queries — you would actually type in to find that answer, and for each one a short note on why you'd search it and how it helps answer the question.

Search the way a thoughtful person would: clear, effective queries. Use as many or as few as you genuinely need — no more, no fewer."""

    user = f"Question: {question}"
    return [SystemMessage(content=system), HumanMessage(content=user)]


# --- Stage 2: link picker (structured output) -------------------------------


class LinkPick(BaseModel):
    """One picked URL plus the reason it's worth fetching."""

    url: str = Field(
        description=(
            "The URL to fetch. MUST be copied verbatim from one of the "
            "search results shown in the user message. Do not invent or "
            "modify URLs."
        )
    )
    reason: str = Field(
        description=(
            "One short sentence (max ~20 words) explaining why this URL is "
            "likely to contain a substantive answer to the question. Mention "
            "authority, recency, or specificity."
        )
    )


class LinkPicks(BaseModel):
    """The link-picker's chosen URLs.

    The exact number of picks is specified per call in the system message
    (parameterized via `n_picks` in `build_link_picker_messages`). The
    schema's `max_length` is a hard upper bound so a confused model can't
    emit dozens of picks; the prompt is what tells it the target count.
    """

    picks: list[LinkPick] = Field(
        min_length=1,
        max_length=10,
        description=(
            "The most useful search results, ordered by usefulness (best "
            "first). How many to pick is specified in the system message — "
            "return that many if at all possible. Prefer authoritative and "
            "recent sources; avoid duplicates and near-duplicates."
        )
    )


_LINK_PICKER_EXAMPLE = """{
  "picks": [
    {
      "url": "https://www.langchain.com/langgraph",
      "reason": "Official source from LangChain; most authoritative description."
    },
    {
      "url": "https://github.com/langchain-ai/langgraph",
      "reason": "Official repo with current README and recent release notes."
    }
  ]
}"""


def _format_search_result(idx: int, r: SearchResult) -> str:
    parts = [f"[{idx}] Title: {r.title}", f"    URL: {r.url}"]
    if r.published_date:
        parts.append(f"    Published: {r.published_date.isoformat()}")
    if r.snippet:
        parts.append(f"    Snippet: {r.snippet}")
    return "\n".join(parts)


def build_link_picker_messages(
    question: str,
    results: list[SearchResult],
    n_picks: int = 3,
    now: datetime | None = None,
) -> list:
    """System+user messages for choosing which search results to fetch.

    Uses structured output: returned message validates against `LinkPicks`.
    The schema is also rendered into the system prompt (semantics) alongside
    a worked example (shape).
    """
    n = now or _now_utc()
    schema_block = pydantic_to_prompt(LinkPicks)

    system = f"""{_datetime_context_block(n)}

You are a research assistant deciding which web pages are worth fetching to answer a user's question. You will be shown the question and a numbered list of search results (title, URL, snippet, optional published date). Choose {n_picks} results most likely to contain a substantive, trustworthy answer.

Selection criteria:
- Prefer authoritative sources (official sites, primary documentation, well-known publications) over SEO listicles.
- For time-sensitive questions, prefer recent content; deprioritize results older than 2 years unless they're foundational.
- Pick variety — avoid two near-duplicate sources.
- Each picked URL MUST be copied verbatim from the search results above. Do not invent URLs.

Output schema (JSON):

{schema_block}

Example output:
{_LINK_PICKER_EXAMPLE}
"""

    formatted_results = "\n\n".join(
        _format_search_result(i + 1, r) for i, r in enumerate(results)
    )
    user = f"""Question: {question}

Search results:
{formatted_results}"""

    return [SystemMessage(content=system), HumanMessage(content=user)]


# --- Stage 3: answer with citations (structured output) ---------------------


class CitationRef(BaseModel):
    """One retrieved chunk backing one prose chunk: chunk id + supporting excerpts.

    v1 change from v0: this now identifies the *retrieved chunk* the
    evidence came from, not a source-URL ordinal. Each retrieved chunk
    has its source URL bound at retrieval time, so the answer renderer
    looks up the URL from `chunk_id` deterministically — the model can
    no longer cite the wrong URL because it doesn't write URLs at all.
    This is the structural fix for the cross-source misattribution
    failure mode (known_issues.md).

    Lives INSIDE an `AnswerChunk`. Each chunk's `citations` lists each
    `chunk_id` AT MOST ONCE (enforced by a validator on `AnswerChunk`).
    Multiple verbatim excerpts from the same retrieved chunk live in
    `quotes: list[str]` rather than as separate `CitationRef`s.
    """

    chunk_id: int = Field(
        description=(
            "The id of the retrieved chunk being cited. MUST equal the "
            "integer shown in the chunk header in the user message — "
            "headers look like `[chunk 1 | source: example.com]`. Emit "
            "ONLY the integer (e.g. `1`, not `\"chunk 1\"`). Ids are "
            "small integers 1..N where N is the number of chunks shown."
        )
    )
    quotes: list[str] = Field(
        description=(
            "One or more short verbatim excerpts (each max ~200 chars) "
            "from this chunk's text that support the claim in the prose "
            "chunk's `text`. If multiple distinct quotes from this "
            "retrieved chunk support the claim, include them ALL here in "
            "a single CitationRef — do NOT create separate CitationRefs "
            "for the same chunk_id. Each quote should be a literal copy, "
            "preserving punctuation and capitalization."
        )
    )


class Citation(BaseModel):
    """Downstream citation: one entry per unique source number, grouping
    all the distinct quotes that source backed across all chunks.

    `url` is filled in by `answer_node` from the source numbering used in
    the prompt. Used by run.py, eval, and the trace report. This is the
    per-URL BIBLIOGRAPHY view; for the specific quote behind a specific `[n]`
    on a specific sentence, use `Answer.placements` (per-claim, not per-URL).
    """

    n: int
    url: str
    quotes: list[str]


class PlacedSource(BaseModel):
    """One source `[n]` sitting on one sentence, with the SPECIFIC quote(s) that
    back THAT sentence — the per-claim evidence, not the source's whole quote bag."""

    n: int
    url: str
    quotes: list[str] = Field(default_factory=list)


class Placement(BaseModel):
    """One annotated sentence: the `[n]` markers on it and, per marker, the exact
    supporting quote(s) for this sentence. This is what makes each `[n]` carry the
    sentence-specific evidence instead of the source's aggregated quotes."""

    line: str
    markers: list[int]
    evidence: list[PlacedSource] = Field(default_factory=list)


class Answer(BaseModel):
    """Downstream answer shape (draft-then-cite).

    `text` is the freeform draft annotated in place with inline `[n]` markers
    (see `cite.annotate_in_place`). `citations` is the flattened, group-by-URL
    bibliography for run.py / eval / the trace report. `placements` is the
    per-sentence, per-marker evidence (each `[n]` on a line carries the specific
    quote that backs that line). `claims` holds the grounding review
    (`ClaimAttribution` items) so downstream tools can see exactly which excerpt
    supported which claim — the transparency the old raw `chunks` gave.
    """

    text: str
    citations: list[Citation]
    placements: list[Placement] = Field(default_factory=list)
    # Holds ClaimAttribution items from the attribution pass. Typed as a bare
    # list to avoid a forward-ref on ClaimAttribution, which is defined below.
    claims: list = Field(default_factory=list)


def _format_chunk(local_id: int, domain: str, text: str) -> str:
    """Render one retrieved chunk for the answer prompt.

    Header format is the single-source-of-truth for what the model copies
    into `CitationRef.chunk_id`. The id is a small integer (1..N) assigned
    per-prompt — the storage-layer hex content-hash ID is kept in
    `RetrievalHit.chunk_id` for Chroma dedup, and Python maps the
    prompt-local integer back to the storage hex (and then to the URL)
    at render time.

    Why small integers instead of hex IDs in the prompt: small models
    occasionally botch copying a 16-char hex string (we observed
    `"chunk a3f9c1b2"` literal-prefix bugs). Integers are a single token,
    pydantic's `int` type rejects malformed values structurally, and
    the misattribution-prevention property is preserved because each
    integer maps 1:1 to a unique chunk within this prompt.

    The header shows the **domain** rather than the full URL — the model
    needs source-nature awareness for coherent prose, but the actual URL
    is looked up by Python at render time, so showing the URL would just
    be prompt noise the model might accidentally write into citations.
    """
    return f"[chunk {local_id} | source: {domain}]\n{text}"


# --- Stage 3 (freeform): unstructured baseline, NO citation apparatus --------
# A deliberately unshackled baseline for reading only. The model is given the
# question and the source texts as plain prose — no chunk numbers, no schema,
# no structured output, no citation contract — and asked for a free-form
# factual answer grounded in those sources. The point is to see the ceiling on
# the model's synthesis prose when it is NOT spending capacity on the citation
# machinery the other two variants require. It returns plain messages meant to
# be sent to the raw chat model (`.invoke`), not `with_structured_output`.


DEFAULT_FREEFORM_DIRECTIVES = (
    "Write a clear, factual answer in ordinary prose. Draw on the breadth of "
    "what the sources say and synthesize across them rather than restating each "
    "in turn. Do NOT add citation markers, source numbers, or bracketed "
    "references of any kind — just write the answer."
)


def build_answer_messages_freeform(
    question: str,
    retrieved: list,  # duck-typed: needs .text, .domain
    *,
    critique_feedback: str | None = None,  # accepted for signature parity; ignored
    now: datetime | None = None,
    answer_directives: str | None = None,  # None -> DEFAULT_FREEFORM_DIRECTIVES
) -> list:
    """Free-form, citation-free answer prompt (reading baseline only).

    Same frozen chunks as the other variants, stripped of the citation
    apparatus: sources are shown as plain text labelled only by domain (no
    per-chunk numbers), and the model is asked for a clear factual answer in
    ordinary prose. No schema, no `chunk_id`s, no inline markers. Send the
    returned messages to the raw model (`llm.invoke`) — there is nothing to
    parse. `critique_feedback`, when provided, carries the grounding reviewer's
    list of unsupported claims into a REVISION block — the regeneration hook for
    the draft-then-cite loop (draft freeform, attribute, re-draft on flags).
    """
    n = now or _now_utc()
    directives = answer_directives or DEFAULT_FREEFORM_DIRECTIVES

    system = f"""{_datetime_context_block(n)}

You are a research assistant. Answer the user's question using ONLY the information in the sources provided below. Do NOT add outside knowledge, and do NOT invent facts to fill gaps — if the sources do not answer the question, say so plainly.

{directives}"""

    sources = "\n\n".join(
        f"(source: {c.domain})\n{c.text}" for c in retrieved
    )

    feedback_block = ""
    if critique_feedback:
        feedback_block = f"""

REVISION — a reviewer flagged the following issues with your previous answer:
{critique_feedback}

Write a NEW answer that fixes these issues: keep everything the sources genuinely support, remove or correct any claim flagged as unsupported (do not restate it, and add no new claim the sources do not back), and make sure you fully address every part of the question."""

    user = f"""Question: {question}

Sources:
{sources}{feedback_block}"""

    return [SystemMessage(content=system), HumanMessage(content=user)]


# --- Stage 3c: attribution / grounding review -------------------------------
# Reviewer pass that takes an ALREADY-WRITTEN answer plus the source chunks and
# maps each factual claim back to its supporting chunk(s), flagging any claim no
# chunk supports. This is the "critique as grounding verifier" experiment: it is
# what makes a draft-then-cite flow possible (draft freeform, then attribute),
# and its `supported=false` flags are the hook a regeneration loop would act on.
# Tested in isolation first (eval/attribution_playground.py) — no loop yet.


class ClaimAttribution(BaseModel):
    """One factual claim lifted from the answer, mapped to its evidence.

    `supported` is the flag the grounding review exists to produce: True when
    at least one provided chunk actually states the claim, False when none do.
    When True, `citations` names the supporting chunk(s) with verbatim quotes;
    when False, `citations` is empty. No uniqueness validator here (unlike
    `AnswerChunk`) — this is a read/measure artifact, and we want to see the
    model's raw judgment, including any inconsistencies, rather than reject it.
    """

    claim: str = Field(
        description=(
            "A single factual claim from the answer, in the order it appears, "
            "copied or closely paraphrased. One clear claim per entry. Skip "
            "pure connective/framing sentences that assert no fact."
        )
    )
    supported: bool = Field(
        description=(
            "True if at least one provided chunk actually states this claim; "
            "False if NO chunk does. False is the flag that matters — a claim "
            "with no basis in the chunks."
        )
    )
    citations: list[CitationRef] = Field(
        default_factory=list,
        description=(
            "The supporting chunk(s) with verbatim quotes when supported=True. "
            "MUST be empty when supported=False."
        ),
    )


class AttributionResult(BaseModel):
    """The grounding review of one answer: every factual claim, attributed."""

    claims: list[ClaimAttribution] = Field(
        description=(
            "Every factual claim in the answer, in order, each either "
            "attributed to supporting chunk(s) or flagged unsupported."
        )
    )


_ATTRIBUTION_EXAMPLE = """{
  "claims": [
    {
      "claim": "Qdrant is written in Rust.",
      "supported": true,
      "citations": [
        {"chunk_id": 3, "quotes": ["Qdrant is a high-performance vector similarity search engine and database written in Rust"]}
      ]
    },
    {
      "claim": "Qdrant includes a built-in 5 GB free tier.",
      "supported": false,
      "citations": []
    }
  ]
}"""


def build_attribution_messages(
    question: str,
    answer_text: str,
    retrieved: list,  # duck-typed: needs .domain, .text (same as the answer prompts)
    *,
    now: datetime | None = None,
) -> list:
    """System+user messages for the grounding-review (attribution) pass.

    Given the question, an already-written `answer_text`, and the numbered
    source chunks, the model breaks the answer into factual claims and, for
    each, either cites the supporting chunk(s) with verbatim quotes or flags it
    `supported=false`. The chunk numbering (1..N) is assigned here over
    `retrieved`, identically to the answer builders, so `chunk_id` resolves the
    same way. Complex nested schema, so the in-prompt schema + a worked example
    are kept (per the schema finding: load-bearing for the answer-family
    schemas, and examples beat rules on this model).
    """
    n = now or _now_utc()
    schema_block = pydantic_to_prompt(AttributionResult)

    system = f"""{_datetime_context_block(n)}

You are a strict grounding reviewer. You are given a user's QUESTION, an ANSWER that was already written, and the numbered SOURCE CHUNKS the answer was supposed to be based on. Your ONLY job is to check whether each factual claim in the ANSWER is actually supported by those chunks.

How to review:
- Break the ANSWER into its individual factual claims — one clear claim per entry, in the order they appear. Ignore pure connective or framing sentences that assert no specific fact.
- For each claim, decide whether the provided chunks actually state it:
  - If YES: set `supported` true and cite every supporting chunk by its id, each with one or more VERBATIM quotes copied from that chunk.
  - If NO chunk states it: set `supported` false with an empty `citations` list. This is the flag that matters — a claim with no basis in the chunks.
- Judge ONLY against the chunks. Do NOT use outside knowledge, and do NOT give a claim the benefit of the doubt because it sounds plausible.
- A claim is supported only if a chunk actually states it — NOT merely because a chunk is about the same topic.
- `chunk_id` is the integer in the chunk header (headers look like `[chunk 2 | source: example.com]`). Emit ONLY the integer, 1..N.

Examples:

Supported — a chunk states it, quoted verbatim:
  {{"claim": "Qdrant is written in Rust.", "supported": true,
    "citations": [{{"chunk_id": 3, "quotes": ["... written in Rust"]}}]}}

Unsupported — no chunk states it, so it is flagged:
  {{"claim": "Qdrant includes a built-in 5 GB free tier.", "supported": false, "citations": []}}

Output schema (JSON):

{schema_block}

Example output:
{_ATTRIBUTION_EXAMPLE}
"""

    formatted_chunks = "\n\n".join(
        _format_chunk(i + 1, c.domain, c.text) for i, c in enumerate(retrieved)
    )

    user = f"""Question: {question}

Answer to review:
{answer_text}

Source chunks:
{formatted_chunks}"""

    return [SystemMessage(content=system), HumanMessage(content=user)]


# --- Stage 3d: relevance review (single-job critique) -----------------------
# The narrowed critique for draft-then-cite. Grounding and disclaim are owned by
# the attribution pass now, so this reviewer does the ONE thing attribution
# can't: judge whether the answer actually addresses the question (relevance /
# completeness). Deliberately lean — a grounded answer can still miss a facet or
# drift, and that is all this checks. Needs only (question, answer): the source
# chunks are irrelevant to relevance, so they are not passed in.


class Judgment(BaseModel):
    """One criterion's verdict. `reason` is declared BEFORE `issue_found` so the
    model justifies against the ANSWER/SOURCES first, then commits the boolean —
    keeping the flag consistent with its own reasoning.

    The boolean is named `issue_found` (not `passed`) on purpose: each criterion
    detects a PROBLEM, so TRUE = the problem is present. A field named `passed`
    invited the model to read TRUE as "the answer is fine", inverting the
    intended polarity — worse on stronger models that take the field name
    literally."""

    reason: str = Field(
        description="Brief justification, grounded in the ANSWER and SOURCES."
    )
    issue_found: bool = Field(
        description="TRUE if this criterion's PROBLEM is present in the ANSWER, "
                    "FALSE if the answer is clean on this criterion — see the "
                    "criterion's own description for what the problem is."
    )


class Critique(BaseModel):
    """Two problem-detection judgments about the ANSWER vs the SOURCES — each a
    reason + true/false, where TRUE means the problem is present. The answer is
    sound only if NEITHER problem is found. Sufficiency (whether the sources CAN
    answer at all) is intentionally NOT judged here; that is the answer /
    sufficiency node's job."""

    fabricates_facts: Judgment = Field(
        description=(
            "Does the ANSWER state facts that are not found in the SOURCES, or "
            "that contradict them? Truthfully saying the sources lack some "
            "information is NOT fabrication. TRUE means it fabricates."
        )
    )
    contradicted_by_sources: Judgment = Field(
        description=(
            "Does the ANSWER assert something drawn from one source while "
            "another source gives directly conflicting evidence that the answer "
            "ignores? Sources giving slightly different figures for the same "
            "thing, which the answer reasonably synthesizes, is NOT a "
            "contradiction. TRUE means the answer took one source at face value "
            "despite a material conflict in another."
        )
    )

    @property
    def answer_is_sound(self) -> bool:
        """Sound only if neither problem is present."""
        return (not self.fabricates_facts.issue_found
                and not self.contradicted_by_sources.issue_found)

    @property
    def reason(self) -> str:
        """Regeneration feedback composed from the problems found."""
        parts = []
        if self.fabricates_facts.issue_found:
            parts.append(f"[fabrication] {self.fabricates_facts.reason}")
        if self.contradicted_by_sources.issue_found:
            parts.append(f"[contradiction] {self.contradicted_by_sources.reason}")
        return " ".join(parts)


def build_critique_messages(
    question: str,
    answer_text: str,
    retrieved: list,  # duck-typed: needs .text, .domain
    *,
    now: datetime | None = None,
) -> list:
    """System+user messages for the rubric critique.

    The model answers narrow yes/no criteria about the ANSWER vs QUESTION vs
    SOURCES; `Critique.answer_is_sound` derives the verdict from them in code.
    General-purpose — no worked examples / no dataset-shaped cases: each
    criterion is a plain question the model judges independently.
    """
    n = now or _now_utc()

    system = f"""{_datetime_context_block(n)}

## Context
A user asked a question. An automated web search fetched several web pages, and passages (the SOURCES below) were extracted from them. Another model then wrote the ANSWER using only those sources. The sources are imperfect internet content — they may be partial, may discuss closely related rather than exact entities, and may not cover everything asked.

## Task
Each criterion below describes a possible PROBLEM with the ANSWER. For each, first write a brief reason grounded in the ANSWER and SOURCES, then answer true if the problem is present or false if it is not. Judge only against the SOURCES, not outside knowledge.

## Criteria
- fabricates_facts: Does the ANSWER state facts that are not found in the SOURCES, or that contradict them? Truthfully saying the sources lack some information is not fabrication. True = it fabricates.
- contradicted_by_sources: Does the ANSWER assert something drawn from one source while another source gives directly conflicting evidence that the answer ignores? Sources giving slightly different figures for the same thing, which the answer reasonably synthesizes, is not a contradiction. True = it took one source at face value despite a material conflict in another.

## Notes
Judge on substance, not presentation: do not penalize imperfect structure, minor omissions the answer reasonably left out, or reasonable synthesis across sources."""

    sources = "\n\n".join(
        f"(source: {c.domain})\n{c.text}" for c in retrieved
    )
    user = f"""QUESTION:
{question}

ANSWER:
{answer_text}

SOURCES:
{sources}"""

    return [SystemMessage(content=system), HumanMessage(content=user)]


# --- Sufficiency judge (v1.1: LLM alternative to the score heuristic) -------
#
# `sufficiency.py` decides "can we answer over these chunks?" from cross-encoder
# scores alone. Measured on the frozen corpus, that signal is not merely
# mis-tuned but INVERTED for this decision: stress-disclaim-001 ("Anthropic's Q3
# 2025 revenue?"), whose sources genuinely lack the answer, scores a 5.31 median
# — nearly DOUBLE stress-multi-001's 2.79, which is fully answerable. No
# threshold can separate them, because a cross-encoder scores TOPICAL SIMILARITY,
# not answer presence: chunks about Anthropic's revenue match "Anthropic's Q3
# revenue" beautifully while containing no Q3 figure.
#
# So the judge reads the chunks and answers the question the scores cannot.
# `sufficiency.py`'s docstring anticipated exactly this ("add an LLM judge as
# v1.1 — same interface, different implementation").


class SufficiencyJudgment(BaseModel):
    """Whether the SOURCES actually contain what the QUESTION asks for.

    `reason` precedes the boolean so the model commits only after justifying
    against the sources — same ordering as `Judgment`, for the same reason.

    The boolean is named `sources_answer_question` and is POSITIVE (true =
    good), unlike `Judgment.issue_found`. That is deliberate: this is a single
    plain question, not a problem-detection criterion, so the natural English
    reading and the code meaning already agree. Naming it after a problem would
    reintroduce the polarity trap that inverted the critique on stronger models.
    """

    reason: str = Field(
        description="Brief justification, grounded in the SOURCES. Name what is "
                    "present or what is missing."
    )
    sources_answer_question: bool = Field(
        description="TRUE if the sources contain what the question asks for "
                    "(even partially, enough for a useful answer). FALSE if "
                    "they are merely on-topic without containing it."
    )


#: Default TRUE/FALSE guidance for the sufficiency judge. Swappable via
#: `build_sufficiency_judge_messages(..., guidance=...)` so variants can be
#: A/B'd without forking the whole prompt. Still example-free: principles only.
SUFFICIENCY_GUIDANCE_DEFAULT = """Answer TRUE if the sources contain the information asked for — including when they cover it only partially, or when the answer must be pieced together across several sources. Partial but real evidence is enough.

Answer FALSE when the sources are merely adjacent: about the right subject, entity, or period, but silent on the specific thing asked. A question about one quarter is not answered by annual totals; a question about one product's limit is not answered by a different product's limit. If you cannot point to where the answer is, it is not there."""


def build_sufficiency_judge_messages(
    question: str,
    retrieved: list,  # duck-typed: needs .text, .domain
    *,
    guidance: str | None = None,
    now: datetime | None = None,
) -> list:
    """System+user messages for the LLM sufficiency judge.

    Deliberately example-free: framing and one plain question, no dataset-shaped
    cases that would overfit to our eval set.

    `guidance` swaps the TRUE/FALSE paragraphs (defaults to
    `SUFFICIENCY_GUIDANCE_DEFAULT`); everything else is held constant so a
    variant sweep isolates that wording.
    """
    n = now or _now_utc()

    system = f"""{_datetime_context_block(n)}

## Context
A user asked a question. An automated web search fetched several web pages, and passages (the SOURCES below) were extracted from them. Before we try to answer, we need to know whether these sources actually contain what was asked. The sources are imperfect internet content: they were selected because they are TOPICALLY similar to the question, so they will all look roughly on-topic. Being about the right subject is not the same as containing the answer.

## Task
Read the SOURCES and decide whether they contain what the QUESTION specifically asks for. First write a brief reason naming what is present or missing, then answer true or false. Judge only against the SOURCES, not outside knowledge.

{guidance or SUFFICIENCY_GUIDANCE_DEFAULT}"""

    sources = "\n\n".join(f"(source: {c.domain})\n{c.text}" for c in retrieved)
    user = f"""QUESTION:
{question}

SOURCES:
{sources}"""

    return [SystemMessage(content=system), HumanMessage(content=user)]


# --- Sufficiency judge, verdict form (v3) -----------------------------------
#
# WHY A SECOND SHAPE. The boolean judge above asks a FACT question ("do the
# sources contain the answer?") and leaves the TRUE/FALSE calibration to a
# swappable guidance block. Reading the v0/v1/v2 sweep showed the guidance is
# where the failures live, and that they are not fixable by more guidance:
#
#   v0/v2  refuse "1M on all paid plans" as an answer about the Pro plan --
#          over-literal, no passage says "Pro" -- so they re-search a question
#          whose answer is sitting in Anthropic's own support doc.
#   v1     fixes that, then states "they do not explicitly rank ... which one
#          is 'most actively maintained'" and returns TRUE anyway -- its own
#          reason refutes its verdict.
#
# So this form changes the framing rather than the wording. Two differences:
#
#   1. It asks for a ROUTING DECISION, not a fact. The judge gates a loop; the
#      real question is "answer from these, or search again?". The verdicts are
#      named after the property being judged (`sufficient`/`insufficient`) and
#      each states the CONSEQUENCE that follows from it, so the model is
#      deciding with the downstream cost in view.
#   2. The verdict is a Literal, not a bool. Named outcomes cannot invert the
#      way `Judgment.passed` did on E4B -- there is no English reading of "sufficient" that
#      flips its meaning, whereas "pass" reads as both approve and decline.
#
# Deliberately guidance-free and example-free: no TRUE/FALSE rubric, no
# contrast pairs. The earlier rubrics ("a question about one quarter is not
# answered by annual totals") were shaped by our own eval questions, which is
# exactly the overfitting the no-worked-examples decision exists to avoid.


class SufficiencyVerdict(BaseModel):
    """Routing decision: answer from these sources, or search again?

    `reason` precedes `verdict` so the model commits only after justifying
    against the sources -- same ordering, same rationale, as `Judgment` and
    `SufficiencyJudgment`.

    `verdict` is a Literal rather than a bool on purpose. `SufficiencyJudgment`
    dodged the polarity trap by naming its bool positively; this dodges it by
    having no bool to invert.
    """

    reason: str = Field(
        description="Brief justification, grounded in the SOURCES. Name what is "
                    "present or what is missing."
    )
    verdict: typing.Literal["sufficient", "insufficient"] = Field(
        description="'sufficient' to answer from these sources alone; "
                    "'insufficient' to run a further internet search first."
    )

    @property
    def sources_answer_question(self) -> bool:
        """Bool view, so callers can treat this like `SufficiencyJudgment`."""
        return self.verdict == "sufficient"


def build_sufficiency_verdict_messages(
    question: str,
    retrieved: list,  # duck-typed: needs .text, .domain
    *,
    now: datetime | None = None,
) -> list:
    """System+user messages for the verdict-form sufficiency judge.

    Same `## Context` block as `build_sufficiency_judge_messages` (it is doing
    its job: it tells the model the sources were selected for TOPICAL
    similarity, which is the whole trap). The `## Task` section is replaced by
    the two consequence-carrying verdicts, and there is no guidance block --
    see the comment above for why.
    """
    n = now or _now_utc()

    system = f"""{_datetime_context_block(n)}

## Context
A user asked a question. An automated web search fetched several web pages, and passages (the SOURCES below) were extracted from them. Before we try to answer, we need to know whether these sources actually contain what was asked. The sources are imperfect internet content: they were selected because they are TOPICALLY similar to the question, so they will all look roughly on-topic. Being about the right subject is not the same as containing the answer.

## Task
Your task is to judge whether the SOURCES are sufficient to answer the QUESTION, and return one of two verdicts:

- `sufficient` — we do not need to further research the internet for more info to give a correct answer to the user; the sources are sufficient, we can answer from them.
- `insufficient` — the sources are not sufficient; a further internet search is needed.

Explain the judgment you made. Judge only against the SOURCES, not outside knowledge."""

    sources = "\n\n".join(f"(source: {c.domain})\n{c.text}" for c in retrieved)
    user = f"""QUESTION:
{question}

SOURCES:
{sources}"""

    return [SystemMessage(content=system), HumanMessage(content=user)]
