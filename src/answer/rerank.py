"""Cross-encoder reranking via sentence-transformers.

Two-stage retrieval: the first stage (dense embedding cosine) finds K
candidates that are *approximately* relevant — fast, but the cosine of
two 768-dim summaries is a coarse signal. A cross-encoder reads the
`(query, chunk)` pair as one input and lets a transformer attend across
both texts together. The relevance score is much sharper. It can't be
precomputed and indexed (the score depends on the query, so nothing to
pre-store), so we run it only over the K candidates the first stage
already shortlisted.

Model:
- `cross-encoder/ms-marco-MiniLM-L-12-v2` — 33M params, 512-token
  context. The well-known baseline for general English QA reranking,
  small enough to run on CPU in ~1s at K=40.

The model is lazy-loaded on first call. Loading takes ~3s and pulls
the weights from HuggingFace Hub on the very first run (~50 MB). On
subsequent runs the local HF cache makes it instant.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from .vector_store import RetrievalHit

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    """Lazy-load and cache the cross-encoder. First call pays ~3s of init
    plus ~50 MB of weight download on the very first run; subsequent calls
    in the same process are free. We don't preload at import time because
    that would slow eval startup and obscure import-side errors with a
    model-load traceback.
    """
    return CrossEncoder(_MODEL_NAME)


def rerank(query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """Populate each hit's `rerank_score` and sort descending.

    The list is **mutated in place** (hits' `rerank_score` is filled) AND
    returned sorted by `rerank_score` descending. Both behaviors are
    intentional — the mutation makes the score visible at every downstream
    inspection point (state, trace JSON), and the return value makes the
    caller's intent obvious at the call site.

    Batched predict: all K (query, text) pairs go through `.predict()`
    in one call. Running them one at a time costs ~10× more for K=40
    because each call pays the same model-overhead.
    """
    if not hits:
        return []
    pairs = [(query, h.text) for h in hits]
    scores = _model().predict(pairs, batch_size=32, show_progress_bar=False)
    for h, s in zip(hits, scores):
        h.rerank_score = float(s)
    hits.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
    return hits
