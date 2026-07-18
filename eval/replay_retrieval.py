"""Retrieval stage, replayed offline: question-level vs per-QUERY selection.

The first stage we RUN rather than freeze. Everything upstream (queries, search,
pages, chunks, embeddings, query embeddings) is frozen, so this is pure local
math plus the local cross-encoder -- no endpoint, no network, infinitely
repeatable, and identical input on every run.

THE QUESTION IT EXISTS TO ANSWER. The `needs` planner decomposes a question into
distinct information needs, one query per need... and then production throws that
structure away: it retrieves ONCE with the QUESTION embedding over the merged
corpus and takes a global top-8. Nothing guarantees each facet is represented, so
a facet whose chunks score lower can be crowded out entirely -- and then the
answer stage honestly reports it could not cover everything asked. Observed in
the wild: stress-multi-002 ("architectures, pricing, and benchmark performance")
answered "I couldn't find a single source that directly compares ... all in one
place".

  --mode question   ONE retrieval with the question embedding (production shape)
  --mode per-query  ONE retrieval PER frozen query, each with its own budget

BUDGET (`--total`, default 12): per_query = max(3, total // n_queries).

    1q -> 12    2q -> 6    3q -> 4    4q -> 3      (total holds at 12)
    5q -> 3     6q -> 3                            (floor wins; total grows)

Both modes get the SAME total for n<=4, which is what makes the A/B honest: if
per-query simply got more chunks it would "win" by feeding the model more text
rather than by selecting better. The floor of 3 is deliberate above 4 queries --
a facet with 2 chunks is effectively unrepresented, which is the starvation this
mode exists to fix, so there it is worth spending the context. Nothing in the
current queries.json exceeds 3 queries, so the floor never fires today.

NOTE both modes use total=12 while production's TOP_N_ANSWER is 8. That keeps
the A/B apples-to-apples, but it means neither arm is production's chunk count --
"12 vs 8" is a separate knob, not something this tool answers.

REAL CODE PATH: uses the actual `rerank()` and `select_mmr()`. Only Chroma's
`similarity_search` is replaced -- by a dot product over the frozen vectors,
which is exactly equivalent here since every vector is L2-normalized (verified:
all norms == 1.0), so cosine == dot.

JSON is the contract, markdown is the view: the selection lands in
`eval/fixtures/selection_<mode>.json` for the answer stage to consume, and a
readable report goes to `eval/reports/`.

Needs the local cross-encoder (torch/sentence-transformers -- the slow import),
but NO endpoint.

Usage:
    uv run python eval/replay_retrieval.py --mode question
    uv run python eval/replay_retrieval.py --mode per-query
    uv run python eval/replay_retrieval.py --mode per-query --only stress-multi-002
    uv run python eval/replay_retrieval.py --mode per-query --total 8
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from answer.config import MMR_LAMBDA, TOP_K_RETRIEVAL  # noqa: E402
from answer.mmr import select_mmr  # noqa: E402
from answer.rerank import rerank  # noqa: E402
from answer.vector_store import RetrievalHit  # noqa: E402

FIXTURES = _EVAL_DIR / "fixtures"
REPORTS = _EVAL_DIR / "reports"
DEFAULT_TOTAL = 12
MIN_PER_QUERY = 3


def _load(name: str) -> dict:
    p = FIXTURES / name
    if not p.exists():
        print(f"missing {p.relative_to(_PROJECT_ROOT)} — run the freeze chain first")
        sys.exit(2)
    return json.loads(p.read_text(encoding="utf-8"))


def budget(n_queries: int, total: int) -> int:
    """Chunks per query. Floor of 3 so a facet is never token-starved."""
    return max(MIN_PER_QUERY, total // n_queries)


def search_offline(qvec: list[float], pool: list[RetrievalHit],
                   k: int) -> list[RetrievalHit]:
    """Chroma's similarity_search, offline. Vectors are L2-normalized, so the
    dot product IS cosine — no normalization step needed."""
    scored = []
    for h in pool:
        s = sum(a * b for a, b in zip(qvec, h.embedding))
        scored.append(
            RetrievalHit(
                chunk_id=h.chunk_id, text=h.text, url=h.url, domain=h.domain,
                position=h.position, retrieval_score=s, embedding=h.embedding,
            )
        )
    scored.sort(key=lambda h: -h.retrieval_score)
    return scored[:k]


def run_one(text: str, qvecs: dict, pool: list[RetrievalHit],
            top_n: int) -> tuple[list[RetrievalHit], list[RetrievalHit]]:
    """retrieve -> rerank -> MMR, the real code path. Returns (reranked_pool,
    selected): the FULL reranked candidate pool (for the retrieve/rerank funnel)
    and the MMR-selected subset."""
    if text not in qvecs:
        print(f"  !! no frozen query vector for {text[:50]!r} — "
              "re-run freeze_query_embeddings.py")
        sys.exit(2)
    hits = search_offline(qvecs[text], pool, TOP_K_RETRIEVAL)
    reranked = rerank(text, hits)
    selected = select_mmr(reranked, lambda_=MMR_LAMBDA, top_n=top_n)
    return reranked, selected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["question", "per-query"], required=True)
    ap.add_argument("--total", type=int, default=DEFAULT_TOTAL,
                    help=f"chunk budget per question (default {DEFAULT_TOTAL})")
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()

    search = _load("search_results.json")["questions"]
    chunks = _load("chunks.json")["chunks"]
    demb = _load("embeddings.json")
    qemb = _load("query_embeddings.json")

    if demb["embed_model"] != qemb["embed_model"]:
        print(f"MODEL MISMATCH: docs={demb['embed_model']} "
              f"queries={qemb['embed_model']} — different vector spaces, "
              "every score would be silently wrong. Re-freeze one side.")
        sys.exit(2)

    qvecs = qemb["embeddings"]
    dvecs = demb["embeddings"]

    rows = search
    if args.only:
        wanted = {q.strip() for q in args.only.split(",") if q.strip()}
        rows = [r for r in rows if r["qid"] in wanted]
    if not rows:
        print("no questions selected")
        sys.exit(2)

    by_url: dict[str, list[dict]] = {}
    for c in chunks:
        by_url.setdefault(c["url"], []).append(c)

    print(f"mode={args.mode} · total={args.total} · top_k={TOP_K_RETRIEVAL} · "
          f"mmr_lambda={MMR_LAMBDA}")
    print(f"loading cross-encoder (slow first import)…\n", flush=True)

    out_rows = []
    for r in rows:
        qid = r["qid"]
        # ITERATION-AWARE. A question that went round the refinement loop was
        # selected more than once, over a GROWING pool: iteration i sees the URLs
        # from rounds <= i. Recomputing every iteration (not just the latest)
        # keeps the loop's history instead of overwriting it — and because this
        # stage is deterministic over frozen inputs, that history can be rebuilt
        # from scratch at any time. The rounds actually SEARCHED are exactly the
        # rounds present in search_results (a STOP round has no queries, so it
        # contributes no entry and no iteration).
        rounds_present = sorted({e.get("round", 0) for e in r["queries"]})
        iterations = []
        for it in rounds_present:
            entries = [e for e in r["queries"] if e.get("round", 0) <= it]
            urls = {x["url"] for e in entries for x in e["results"]}
            pool = [
                RetrievalHit(chunk_id=c["id"], text=c["text"], url=c["url"],
                             domain=c["domain"], position=c["position"],
                             retrieval_score=0.0, embedding=dvecs[c["id"]])
                for u in urls for c in by_url.get(u, []) if c["id"] in dvecs
            ]
            queries = [e["query"] for e in entries]

            if args.mode == "question":
                reranked, selected = run_one(r["question"], qvecs, pool, args.total)
                groups = [{"query": r["question"], "per_query": args.total,
                           "reranked": reranked, "hits": selected}]
            else:
                per = budget(len(queries), args.total)
                groups = []
                for q in queries:
                    reranked, selected = run_one(q, qvecs, pool, per)
                    groups.append({"query": q, "per_query": per,
                                   "reranked": reranked, "hits": selected})

            # A chunk can top-rank for two queries. Dedup first-wins so the answer
            # stage never sees the same text twice; the total may land under target.
            seen: set[str] = set()
            for g in groups:
                keep = []
                for h in g["hits"]:
                    if h.chunk_id not in seen:
                        seen.add(h.chunk_id)
                        keep.append(h)
                g["hits"] = keep

            n_sel = sum(len(g["hits"]) for g in groups)
            doms = len({h.domain for g in groups for h in g["hits"]})
            dupes = sum(g["per_query"] for g in groups) - n_sel
            print(f"[{qid:22}] iter={it} pool={len(pool):>4} "
                  f"queries={len(queries)} selected={n_sel:>2} domains={doms:>2}"
                  + (f"  ({dupes} cross-query dupes dropped)" if dupes else ""))

            iterations.append({
                "iteration": it,
                "n_selected": n_sel,
                "n_domains": doms,
                "pool_size": len(pool),
                "groups": [
                    {
                        "query": g["query"],
                        "per_query": g["per_query"],
                        # the FULL reranked pool (top-K) for the retrieve/rerank
                        # funnel, each flagged with whether MMR kept it
                        "candidates": [
                            {"id": h.chunk_id, "domain": h.domain, "url": h.url,
                             "retrieval_score": round(float(h.retrieval_score), 4),
                             "rerank_score": (round(float(h.rerank_score), 4)
                                              if h.rerank_score is not None else None),
                             "selected": h.chunk_id in {s.chunk_id for s in g["hits"]}}
                            for h in g["reranked"]
                        ],
                        "chunks": [
                            {"id": h.chunk_id, "domain": h.domain, "url": h.url,
                             "rerank_score": (round(float(h.rerank_score), 4)
                                              if h.rerank_score is not None else None),
                             "retrieval_score": round(float(h.retrieval_score), 4),
                             "text": h.text}
                            for h in g["hits"]
                        ],
                    }
                    for g in groups
                ],
            })

        out_rows.append({
            "qid": qid,
            "question": r["question"],
            "iterations": iterations,
        })

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema": "iterations/v1",
        "mode": args.mode,
        "total": args.total,
        "min_per_query": MIN_PER_QUERY,
        "top_k": TOP_K_RETRIEVAL,
        "mmr_lambda": MMR_LAMBDA,
        "embed_model": demb["embed_model"],
        "questions": out_rows,
    }
    out = FIXTURES / f"selection_{args.mode}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    REPORTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md = REPORTS / f"retrieval_{args.mode}_{ts}.md"
    md.write_text(render(payload), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"selection -> {out.relative_to(_PROJECT_ROOT)}")
    print(f"report    -> {md.relative_to(_PROJECT_ROOT)}")
    print("=" * 60)


def render(p: dict) -> str:
    o = [f"# Retrieval — mode `{p['mode']}`", "",
         f"- Generated: `{p['generated']}`",
         f"- total={p['total']} · min_per_query={p['min_per_query']} · "
         f"top_k={p['top_k']} · mmr_lambda={p['mmr_lambda']}",
         f"- Embeddings: `{p['embed_model']}` (frozen, offline)", ""]
    for r in p["questions"]:
        o.append(f"### `{r['qid']}` — {r['question']}")
        o.append("")
        for itr in r["iterations"]:
            # Multi-iteration questions went round the refinement loop; each
            # iteration saw a larger pool (rounds <= i).
            o.append(f"#### iteration {itr['iteration']}")
            o.append("")
            o.append(f"pool={itr['pool_size']} chunks · selected={itr['n_selected']} · "
                     f"domains={itr['n_domains']}")
            o.append("")
            for g in itr["groups"]:
                o.append(f"**query:** `{g['query']}` _(budget {g['per_query']})_")
                o.append("")
                for c in g["chunks"]:
                    rs = c["rerank_score"]
                    score = (f"rr={rs}" if rs is not None
                             else f"ret={c['retrieval_score']}")
                    o.append(f"- **{c['domain']}** _{score}_")
                    text = c["text"][:300].replace("\n", " ")
                    o.append(f"  > {text}…")
                o.append("")
    return "\n".join(o) + "\n"


if __name__ == "__main__":
    main()
