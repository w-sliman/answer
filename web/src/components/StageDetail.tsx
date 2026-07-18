import type { Question, StageKind, Candidate, SelectionIter } from "../data";
import { RUN } from "../data";

const domainOf = (url: string) => {
  try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return url; }
};

// The full retrieve/rerank pool (top-40) for a selection iteration. Falls back
// to the selected finalists (treated as all-kept) for any older run.json that
// predates the candidates field, so the funnel view degrades gracefully.
function poolOf(sel: SelectionIter | undefined): Candidate[] {
  if (sel?.candidates?.length) return sel.candidates;
  return (sel?.chunks ?? []).map((c) => ({
    id: c.id, domain: c.domain, url: c.url,
    retrieval_score: c.retrieval_score, rerank_score: c.rerank_score, selected: true,
  }));
}

export function StageDetail({ kind, iteration, q }: { kind: StageKind; iteration: number; q: Question }) {
  switch (kind) {
    case "plan": {
      const round = q.plan.find((r) => r.round === 0);
      return (
        <div className="grid2">
          {round?.queries.map((qy, i) => (
            <div className="card" key={i}>
              <div className="qy">{qy.query}</div>
              {qy.why && <div className="why">{qy.why}</div>}
            </div>
          ))}
        </div>
      );
    }
    case "search": {
      const rounds = q.search.filter((s) => s.round === iteration);
      return (
        <>
          {rounds.map((r, ri) => (
            <div className="card" key={ri}>
              <div className="qy" style={{ marginBottom: 10 }}>{r.query}</div>
              {r.results.slice(0, 4).map((res, i) => (
                <div className="res" key={i}>
                  <div className="rt">{res.title || res.url}</div>
                  <div className="rd">{res.domain}</div>
                  {res.snippet && <div className="rs">{res.snippet}</div>}
                </div>
              ))}
            </div>
          ))}
        </>
      );
    }
    case "pick_links": {
      const picked = q.pages ?? [];
      return (
        <div className="card">
          <div className="reason" style={{ marginBottom: picked.length ? 12 : 0 }}>
            An LLM curates the URLs worth reading — here <b>{picked.length}</b> from the{" "}
            {q.funnel.n_unique_urls} unique results, favouring primary sources over aggregators.
          </div>
          {picked.map((p, i) => (
            <div className="res" key={i}>
              <div className="rt">{p.domain}</div>
              <div className="rd">{p.url}</div>
            </div>
          ))}
        </div>
      );
    }
    case "fetch": {
      const pages = q.pages ?? [];
      return (
        <div className="card">
          <div className="reason" style={{ marginBottom: pages.length ? 12 : 0 }}>
            Fetched <b>{q.funnel.n_pages_fetched}</b> of {q.funnel.n_unique_urls} unique URLs to
            markdown. Paywalled or JS-walled pages are recorded as misses and skipped.
          </div>
          {pages.map((p, i) => (
            <div className="res" key={i}>
              <div className="rt">{p.url}</div>
              <div className="rd">{p.domain} · {(p.chars / 1000).toFixed(1)}k chars</div>
            </div>
          ))}
        </div>
      );
    }
    case "index": {
      const sample = q.index_sample ?? [];
      return (
        <div className="card">
          <div className="reason" style={{ marginBottom: sample.length ? 12 : 0 }}>
            Split the pages into <b>{q.funnel.n_chunks}</b> passages ({RUN.meta.chunk_size}-char
            windows, {RUN.meta.chunk_overlap} overlap) and embedded each with{" "}
            <b>{RUN.meta.embed_model}</b> ({RUN.meta.embed_dim}-d) into a vector index.
          </div>
          {sample.map((c, i) => (
            <div className="res" key={i}>
              <div className="rd">{c.domain}</div>
              <div className="rs">{c.text}…</div>
            </div>
          ))}
        </div>
      );
    }
    case "retrieve": {
      const sel = q.selection.find((s) => s.iteration === iteration);
      const pool = poolOf(sel);
      const cands = [...pool].sort((a, b) => (b.retrieval_score ?? 0) - (a.retrieval_score ?? 0));
      const kept = cands.filter((c) => c.selected).length;
      const mx = Math.max(1e-6, ...cands.map((c) => c.retrieval_score ?? 0));
      return (
        <div className="card">
          <div className="reason" style={{ marginBottom: cands.length ? 12 : 0 }}>
            Dense retrieval embeds the question and pulls the top-{cands.length} candidates by cosine
            similarity. All {cands.length} are shown, ordered by cosine — the <b>{kept}</b> that survive
            selection are highlighted; the rest are dropped downstream.
          </div>
          {cands.map((c) => (
            <div className={`rk${c.selected ? "" : " drop"}`} key={c.id}>
              <span className="dom">{c.domain}</span>
              <span className="barwrap">
                <span className="bar" style={{ width: `${Math.max(4, ((c.retrieval_score ?? 0) / mx) * 100)}%` }} />
              </span>
              <span className="sc">{(c.retrieval_score ?? 0).toFixed(2)}</span>
              <span className="keep">{c.selected ? "kept" : ""}</span>
            </div>
          ))}
        </div>
      );
    }
    case "rerank": {
      const sel = q.selection.find((s) => s.iteration === iteration);
      const pool = poolOf(sel);
      const cosRank = new Map(
        [...pool].sort((a, b) => (b.retrieval_score ?? 0) - (a.retrieval_score ?? 0))
          .map((c, i) => [c.id, i] as const),
      );
      const byRerank = [...pool].sort((a, b) => (b.rerank_score ?? 0) - (a.rerank_score ?? 0));
      const kept = pool.filter((c) => c.selected).length;
      const mx = Math.max(1e-6, ...pool.map((c) => c.rerank_score ?? 0));
      return (
        <div className="card">
          <div className="reason" style={{ marginBottom: byRerank.length ? 12 : 0 }}>
            A MiniLM cross-encoder rescores all {byRerank.length} candidates against the question — a
            sharper signal than cosine. <b>↑/↓</b> is how far each moved from its dense rank; the{" "}
            <b>{kept}</b> kept are highlighted.
          </div>
          {byRerank.map((c, i) => {
            const move = (cosRank.get(c.id) ?? i) - i;
            return (
              <div className={`rk${c.selected ? "" : " drop"}`} key={c.id}>
                <span className="dom">{c.domain}</span>
                <span className={`mv ${move > 0 ? "up" : move < 0 ? "down" : ""}`}>
                  {move > 0 ? `↑${move}` : move < 0 ? `↓${-move}` : "–"}
                </span>
                <span className="barwrap">
                  <span className="bar" style={{ width: `${Math.max(4, ((c.rerank_score ?? 0) / mx) * 100)}%` }} />
                </span>
                <span className="sc">{(c.rerank_score ?? 0).toFixed(2)}</span>
                <span className="keep">{c.selected ? "kept" : ""}</span>
              </div>
            );
          })}
        </div>
      );
    }
    case "select": {
      const sel = q.selection.find((s) => s.iteration === iteration);
      const chunks = sel?.chunks ?? [];
      const domains = sel?.n_domains ?? new Set(chunks.map((c) => c.domain)).size;
      return (
        <div className="card">
          <div className="reason" style={{ marginBottom: chunks.length ? 12 : 0 }}>
            MMR (λ=0.7) selects <b>{sel?.n_selected ?? chunks.length}</b> passages across{" "}
            <b>{domains}</b> domain{domains !== 1 ? "s" : ""} — trading rerank relevance against
            diversity so one source can't dominate. These are what the answer is written from.
          </div>
          {chunks.map((c, i) => (
            <div className="res" key={i}>
              <div className="rd">{c.domain}</div>
              <div className="rs">{c.text}</div>
            </div>
          ))}
        </div>
      );
    }
    case "judge": {
      const suff = q.sufficiency.find((s) => s.iteration === iteration);
      if (!suff) return null;
      const ok = suff.sufficient;
      return (
        <div className="card">
          <div className="verdict">
            <span className={`pill ${ok ? "ok" : "no"}`}>● {ok ? "sufficient" : "insufficient"}</span>
            <span className="route">
              route → <b>{ok ? "answer" : "refine"}</b>
            </span>
          </div>
          <p className="reason">{suff.reason}</p>
        </div>
      );
    }
    case "refine": {
      const ref = q.refinement;
      if (!ref) return null;
      return (
        <div className="card">
          <div className="verdict">
            <span className={`pill ${ref.searchable ? "ok" : "no"}`}>
              ● {ref.searchable ? "search again" : "stop — unfillable"}
            </span>
          </div>
          <p className="reason" style={{ marginBottom: ref.queries.length ? 12 : 0 }}>
            {ref.reason}
          </p>
          {ref.queries.map((qy, i) => (
            <div className="qy" key={i} style={{ marginTop: 8 }}>{qy.query}</div>
          ))}
        </div>
      );
    }
    case "answer": {
      const a = q.answer;
      const grounded = (a.n_unsupported ?? 0) === 0;
      const attempts = a.answer_attempts ?? 1;
      return (
        <div className="card">
          <div className="verdict">
            <span className={`pill ${grounded ? "ok" : "no"}`}>
              ● {grounded ? "grounded" : `${a.n_unsupported} unsupported`}
            </span>
            <span className="route">
              {a.n_claims ?? 0} claims · {attempts} attempt{attempts !== 1 ? "s" : ""}
            </span>
          </div>
          <p className="reason">
            Drafted a freeform answer over the selected chunks, then bound every claim to its
            supporting passage. The grounded answer is shown below.
          </p>
        </div>
      );
    }
    case "cite":
      return (
        <div className="card">
          {q.annotation.citations.map((c) => (
            <div className="src" key={c.n}>
              <span className="mk">[{c.n}]</span>
              <span className="u">{domainOf(c.url)}</span>
              <span className="qn">{c.quotes.length} quote{c.quotes.length !== 1 ? "s" : ""}</span>
            </div>
          ))}
        </div>
      );
    default:
      return null;
  }
}
