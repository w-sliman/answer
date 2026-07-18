"""Maximal Marginal Relevance selection.

MMR balances relevance against diversity when picking N items from K
candidates. The greedy algorithm:

  1. Pick the most relevant candidate first.
  2. For each subsequent pick, choose the candidate that maximizes
         λ * relevance(c)  -  (1 - λ) * max_similarity(c, already_picked)
     where similarity uses the dense embedding (cosine).
  3. Stop when N picked.

  λ = 1.0  → pure relevance (no diversity term).
  λ = 0.0  → pure diversity (no relevance term).
  λ = 0.7  → standard default; relevance-leaning but penalizes redundancy.

This module runs MMR *over the rerank output* (not over raw retrieval),
which is the right architectural placement: the cross-encoder gives a
much sharper relevance signal than cosine-to-query, and the embeddings
we already have are the best signal for "are these chunks topically
similar to each other". One MMR call combines both.

Score normalization. Cross-encoder scores are unbounded (typically
roughly in [-10, +10]) while cosine similarities are in [0, 1]. We
min-max normalize the relevance scores across the candidate set so
both terms in the MMR formula share the [0, 1] range. Without that,
λ stops meaning "how much to trade off" and starts meaning "what units
do the relevance and diversity terms happen to live in" — which is a
much worse knob.
"""
from __future__ import annotations

import numpy as np

from .vector_store import RetrievalHit


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize into [0, 1].

    If all values are identical (degenerate — every candidate scored
    equally), return all 1.0s. That collapses MMR to pure diversity for
    this call, which is the right behavior: when the relevance signal
    can't distinguish, diversity is the only signal we have left.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def select_mmr(
    candidates: list[RetrievalHit],
    *,
    lambda_: float,
    top_n: int,
) -> list[RetrievalHit]:
    """Select up to `top_n` hits using MMR.

    Relevance signal is `rerank_score` when present (Phase 2+), falling
    back to `retrieval_score` otherwise — same code path works whether
    rerank is enabled or not. Diversity signal is each hit's `embedding`
    (cosine sim between unit-normalized vectors).

    Returns a new list. The input is not mutated. Items are returned in
    selection order — first pick is the most-relevant, later picks
    trade relevance against distance from already-picked items.

    Edge cases:
    - Empty input: returns [].
    - top_n >= len(candidates): MMR has nothing to do; return all
      candidates sorted by relevance (the most useful fallback ordering).
    """
    if not candidates:
        return []
    if top_n >= len(candidates):
        return sorted(
            candidates,
            key=lambda h: (
                h.rerank_score if h.rerank_score is not None else h.retrieval_score
            ),
            reverse=True,
        )

    # Build the embedding matrix once. Pre-normalize to unit vectors so
    # cosine similarity reduces to a dot product (cheap).
    emb = np.array([h.embedding for h in candidates], dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    emb_norm = emb / norms

    raw_relevance = [
        h.rerank_score if h.rerank_score is not None else h.retrieval_score
        for h in candidates
    ]
    rel = np.array(_normalize(raw_relevance), dtype=np.float32)

    selected_idx: list[int] = []
    remaining = set(range(len(candidates)))

    # First pick: the candidate with the highest (normalized) relevance.
    first = int(np.argmax(rel))
    selected_idx.append(first)
    remaining.discard(first)

    while len(selected_idx) < top_n and remaining:
        sel_emb = emb_norm[selected_idx]            # (S, D)
        rem_idx = list(remaining)
        rem_emb = emb_norm[rem_idx]                  # (R, D)
        # For each remaining candidate r, max similarity to any selected s
        sim = rem_emb @ sel_emb.T                    # (R, S)
        max_sim = sim.max(axis=1)                    # (R,)
        scores = lambda_ * rel[rem_idx] - (1.0 - lambda_) * max_sim
        best_local = int(np.argmax(scores))
        best_global = rem_idx[best_local]
        selected_idx.append(best_global)
        remaining.discard(best_global)

    return [candidates[i] for i in selected_idx]
