"""Assemble the frozen fixtures into ONE playback payload for the demo site.

Data-first, same discipline as the freeze chain: the website is a pure replay, so
its single input is a self-contained `web/data/run.json` combining every stage of
the one frozen run. Model-free, offline, pure stdlib — just a join over the
fixtures. Re-run whenever a fixture changes.

Per question it emits the full stage walk:
  plan (query rounds) → search (results per round) → corpus funnel (pages/chunks
  counts) → selection (the ranked, selected chunks with scores + text) →
  sufficiency (verdict per iteration) → refinement (STOP/SEARCH decision) →
  answer (the grounded draft + attribution) → annotation (cited text + per-[n]
  placements + citations).

Output: web/data/run.json
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

_ROOT = Path(__file__).resolve().parent.parent
FIX = _ROOT / "eval" / "fixtures"
OUT = _ROOT / "web" / "src" / "run.json"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:  # noqa: BLE001
        return url


def main() -> None:
    queries = _load("queries.json")
    search = {q["qid"]: q for q in _load("search_results.json")["questions"]}
    pages = _load("pages.json").get("pages", {})
    chunks = _load("chunks.json")
    selection = {q["qid"]: q for q in _load("selection_question.json")["questions"]}
    sufficiency = _load("sufficiency.json")["questions"]
    answers = _load("answers.json")["questions"]
    annotations = _load("annotations.json")["questions"]
    embeddings = _load("embeddings.json")

    # chunk_id/url index for the per-question corpus funnel
    chunks_by_url: dict[str, int] = Counter(c["url"] for c in chunks["chunks"])

    out_questions = []
    for qrow in queries["questions"]:
        qid = qrow["qid"]
        question = qrow["question"]

        # --- plan: the query rounds (planner + refinement) ---
        plan_rounds = [
            {
                "round": rd["round"],
                "source": rd.get("source", ""),
                "searchable": rd.get("searchable", True),
                "reason": rd.get("reason", ""),
                "queries": rd.get("queries", []),
            }
            for rd in qrow.get("rounds", [])
        ]

        # --- search: results per (round, query) ---
        s = search.get(qid, {})
        search_rounds = [
            {
                "round": e.get("round", 0),
                "query": e["query"],
                "results": [
                    {"title": r.get("title", ""), "url": r.get("url", ""),
                     "domain": _domain(r.get("url", "")), "snippet": r.get("snippet", "")}
                    for r in e.get("results", [])
                ],
            }
            for e in s.get("queries", [])
        ]

        # --- corpus funnel: this question's URLs -> fetched pages -> chunks ---
        urls = []
        seen = set()
        for e in search_rounds:
            for r in e["results"]:
                u = r["url"]
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)
        fetched = [u for u in urls if u in pages]
        funnel = {
            "n_queries": len(search_rounds),
            "n_results": sum(len(e["results"]) for e in search_rounds),
            "n_unique_urls": len(urls),
            "n_pages_fetched": len(fetched),
            "n_chunks": sum(chunks_by_url.get(u, 0) for u in fetched),
        }

        # --- fetched pages: the actual list (for the "Read" stage detail) ---
        pages_out = [
            {"url": u, "domain": _domain(u), "chars": len(pages.get(u, ""))}
            for u in fetched
        ]
        # --- a sample of the indexed passages (for the "Index" stage detail) ---
        _fetched_set = set(fetched)
        index_sample = [
            {"domain": c.get("domain", ""), "url": c.get("url", ""),
             "text": " ".join(c.get("text", "").split())[:220]}
            for c in chunks["chunks"] if c.get("url") in _fetched_set
        ][:8]

        # --- selection: the ranked, selected chunks per iteration ---
        sel = selection.get(qid, {})
        sel_iters = []
        for itr in sel.get("iterations", []):
            chs = [
                {"id": c["id"], "domain": c.get("domain", ""), "url": c.get("url", ""),
                 "rerank_score": c.get("rerank_score"),
                 "retrieval_score": c.get("retrieval_score"),
                 "text": c.get("text", "")}
                for g in itr["groups"] for c in g["chunks"]
            ]
            # the full reranked candidate pool (for the retrieve/rerank funnel):
            # every candidate with both scores + whether MMR kept it
            cands = [
                {"id": c["id"], "domain": c.get("domain", ""), "url": c.get("url", ""),
                 "retrieval_score": c.get("retrieval_score"),
                 "rerank_score": c.get("rerank_score"),
                 "selected": bool(c.get("selected", False))}
                for g in itr["groups"] for c in g.get("candidates", [])
            ]
            sel_iters.append({
                "iteration": itr["iteration"],
                "pool_size": itr.get("pool_size"),
                "n_selected": itr.get("n_selected", len(chs)),
                "n_domains": itr.get("n_domains"),
                "chunks": chs,
                "candidates": cands,
            })

        # --- sufficiency: verdict per iteration ---
        suff_iters = sufficiency.get(qid, {}).get("iterations", [])

        # --- refinement decision (the searchable refinement round, if any) ---
        refinement = None
        for rd in plan_rounds:
            if rd["source"] == "refinement":
                refinement = {"searchable": rd["searchable"], "reason": rd["reason"],
                              "queries": rd["queries"]}

        a = answers.get(qid, {})
        ann = annotations.get(qid, {})

        out_questions.append({
            "qid": qid,
            "question": question,
            "disclaim": a.get("disclaim", False),
            "n_iterations": max([i["iteration"] for i in suff_iters], default=0) + 1,
            "plan": plan_rounds,
            "search": search_rounds,
            "funnel": funnel,
            "pages": pages_out,
            "index_sample": index_sample,
            "selection": sel_iters,
            "sufficiency": suff_iters,
            "refinement": refinement,
            "answer": {
                "disclaim": a.get("disclaim", False),
                "draft": a.get("draft", ""),
                "answer_attempts": a.get("answer_attempts"),
                "n_claims": a.get("n_claims"),
                "n_unsupported": a.get("n_unsupported"),
                "claims": a.get("claims", []),
            },
            "annotation": {
                "text": ann.get("text", ""),
                "placements": ann.get("placements", []),
                "citations": ann.get("citations", []),
                "n_citations": ann.get("n_citations"),
                "n_markers": ann.get("n_markers"),
            },
        })

    payload = {
        "meta": {
            "dataset": queries.get("dataset"),
            "generated": queries.get("generated"),
            "chat_model": queries.get("model"),
            "embed_model": embeddings.get("embed_model"),
            "embed_dim": embeddings.get("dim"),
            "search_backend": _load("search_results.json").get("backend"),
            "chunk_size": chunks.get("chunk_size"),
            "chunk_overlap": chunks.get("chunk_overlap"),
            "n_pages_total": len(pages),
            "n_chunks_total": chunks.get("n_chunks"),
            "n_questions": len(out_questions),
        },
        "questions": out_questions,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(_ROOT)}  ({size_kb:,.0f} KB · "
          f"{len(out_questions)} questions)")
    # quick per-question sanity line
    for q in out_questions:
        loop = f" · {q['n_iterations']} iters" if q["n_iterations"] > 1 else ""
        tag = "DISCLAIM" if q["disclaim"] else "answer"
        print(f"  {q['qid']:24} {tag:8} sel={len(q['selection'][0]['chunks']) if q['selection'] else 0} "
              f"cites={q['annotation']['n_citations']}{loop}")


if __name__ == "__main__":
    main()
