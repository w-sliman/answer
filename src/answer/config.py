"""Centralized v1 pipeline knobs.

Every number that's a tuning target lives here, not buried in node code. This
makes A/B testing one-line: flip a constant, re-run eval, compare report.

The defaults reflect the design decisions from the v1 planning conversation
(see memory: project_answer.md). Anything marked "calibrate on eval" is a
guess that we'll refine once we see real score distributions.
"""
from __future__ import annotations

from pathlib import Path

# --- Pipeline shape ----------------------------------------------------------

#: Max search queries the planner can emit per loop iteration. 1 for factoid
#: questions, 2-3 for comparative/multi-faceted. Hard cap so a confused planner
#: can't fan out to 10 and burn the search budget.
N_QUERIES_MAX = 3

#: How many URLs the link picker chooses per iteration. Wider than v0's 3 —
#: we trust retrieval to curate across a bigger pool rather than picking blind
#: from snippets.
N_LINKS = 10

#: Outer loop cap. iteration 0 = initial pass, 1 = one refinement.
#: Each iteration is ~30s end-to-end so going past 2 starts hurting UX.
#: Inactive in Phase 1 (loop wired in Phase 4).
MAX_ITERATIONS = 2

#: Inner critique-loop cap inside `answer_node`. attempts 0..N-1 are normal
#: answer→critique cycles; after N failures the node falls back to the
#: disclaim path. 2 is a balance between giving the model a real second
#: chance with feedback and bounding worst-case latency (each cycle is
#: ~5-8s of LLM calls).
MAX_CRITIQUE_ATTEMPTS = 2

# --- Chunking ----------------------------------------------------------------

#: Target chunk size in *characters* (≈ 4× tokens). 1600 chars ≈ 400 tokens
#: — see the v1 planning rationale: balances explanatory questions (need
#: multi-clause context) against factoid questions (need precise embedding).
CHUNK_SIZE_CHARS = 1600

#: Overlap in characters. ~50 tokens. Keeps quote-worthy sentences from
#: being split across boundaries.
CHUNK_OVERLAP_CHARS = 200

# --- Retrieval / rerank ------------------------------------------------------

#: Top-K candidates pulled from the vector store per question. Wide enough
#: that we don't trust the embedder's top-K=20; the reranker re-orders these
#: with a sharper relevance signal. K=40 costs ~1s of CPU rerank time vs
#: K=20's ~0.5s, and the wider safety net is the whole point of the two-
#: stage pipeline.
TOP_K_RETRIEVAL = 40

#: Chunks fed to the answer node (after rerank + MMR). 12 on Gemma 4 E4B:
#: 12 × ~400 tokens ≈ 4.8k of evidence + prompt + response fits comfortably.
#: (Was 8 in the E2B era — the frozen run and current pipeline both use 12.)
TOP_N_ANSWER = 12

#: MMR balance — 1.0 = pure relevance (no diversity), 0.0 = pure diversity.
#: 0.7 is the standard default; tune if answers drift incoherent.
MMR_LAMBDA = 0.7

# (The score-based sufficiency floors were deleted with the heuristic — the
# sufficiency check is now the LLM judge in `sufficiency.py`.)

# --- Embedding ---------------------------------------------------------------

#: Which text we embed as the retrieval query. "question" = the user's
#: original natural-language question (recommended). "search_query" = the
#: LLM-rewritten keyword query (tuned for ddgs lexical matching, not dense
#: retrieval). Flag for A/B testing later.
EMBED_QUERY_SOURCE = "question"  # "question" | "search_query"

#: EmbeddingGemma uses asymmetric task prompts. These prefixes are wrapped
#: around document text and query text respectively before embedding.
#: Getting these wrong silently tanks recall ~10-20%, so they live in one
#: place that the embedding wrapper enforces.
EMBED_DOCUMENT_PREFIX_TEMPLATE = "title: none | text: {text}"
EMBED_QUERY_PREFIX_TEMPLATE = "task: search result | query: {text}"

# --- Vector store ------------------------------------------------------------

#: Local file-backed Chroma path. Project-local. Gitignored.
VECTOR_STORE_PATH = Path(__file__).resolve().parents[2] / "vector_store"
