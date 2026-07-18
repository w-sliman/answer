// Folds the pipeline's live SSE events into the SAME `Question` shape the frozen
// run.json uses, plus the ordered Station list — so the live view renders the
// identical StageTree + StageDetail as replay.
import type { Question, StageKind, SelectionIter } from "./data";
import type { Station } from "./components/StageTree";
import type { AnswerPayload } from "./live";

const domainOf = (url: string) => {
  try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return url; }
};

function emptyQuestion(question: string): Question {
  return {
    qid: "live", question, disclaim: false, n_iterations: 1,
    plan: [], search: [],
    funnel: { n_queries: 0, n_results: 0, n_unique_urls: 0, n_pages_fetched: 0, n_chunks: 0 },
    pages: [], index_sample: [],
    selection: [], sufficiency: [], refinement: null,
    answer: { disclaim: false, draft: "", answer_attempts: null, n_claims: null, n_unsupported: null },
    annotation: { text: "", placements: [], citations: [], n_citations: null, n_markers: null },
  };
}

const MAP: Record<string, { kind: StageKind; label: string }> = {
  search: { kind: "search", label: "Search" },
  pick_links: { kind: "pick_links", label: "Pick sources" },
  fetch: { kind: "fetch", label: "Read" },
  chunk_and_index: { kind: "index", label: "Index" },
  retrieve: { kind: "retrieve", label: "Retrieve" },
  rerank: { kind: "rerank", label: "Rerank" },
  select_chunks: { kind: "select", label: "Select" },
  sufficiency_check: { kind: "judge", label: "Judge" },
  answer: { kind: "answer", label: "Answer" },
  cite: { kind: "cite", label: "Cite" },
};

// generate_queries is the planner on iter 0 and the refinement on later passes —
// exactly how the real graph works (there is no separate refine node).
function mapStage(stage: string, iter: number): { kind: StageKind; label: string } {
  if (stage === "generate_queries")
    return iter > 0 ? { kind: "refine", label: "Refine" } : { kind: "plan", label: "Plan" };
  return MAP[stage] ?? { kind: "plan", label: stage };
}

export class LiveRun {
  q: Question;
  stations: Station[] = [];
  private iter = 0;
  private seenPlan = false;
  private seq = 0;

  constructor(question: string) { this.q = emptyQuestion(question); }

  private lastActive(kind: StageKind): Station | undefined {
    for (let i = this.stations.length - 1; i >= 0; i--)
      if (this.stations[i].kind === kind && this.stations[i].state === "active") return this.stations[i];
    return undefined;
  }

  private setSummary(kind: StageKind, summary: string) {
    const st = this.lastActive(kind);
    if (st && summary) st.summary = summary;
  }

  stageStart(stage: string) {
    if (stage === "generate_queries") {
      if (this.seenPlan) this.iter += 1;
      this.seenPlan = true;
    }
    const { kind, label } = mapStage(stage, this.iter);
    // answer/cite are terminal — keep them at iteration 0 like replay (no hot tint)
    const iteration = kind === "answer" || kind === "cite" ? 0 : this.iter;
    this.stations.push({
      key: `${kind}-${iteration}-${this.seq++}`, kind, iteration,
      label, state: "active", summary: "", loopmark: false,
    });
  }

  stageEnd(stage: string) {
    const st = this.lastActive(mapStage(stage, this.iter).kind);
    if (st) st.state = "done";
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  stageData(stage: string, d: any) {
    const it = this.iter;
    switch (stage) {
      case "generate_queries": {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const queries = (d.queries ?? []).map((x: any) => ({ query: x.query, why: x.why }));
        if (it === 0) {
          this.q.plan.push({ round: 0, source: "planner", searchable: true, reason: "", queries });
          this.setSummary("plan", `${queries.length} quer${queries.length === 1 ? "y" : "ies"}`);
        } else {
          const reason = this.q.sufficiency[this.q.sufficiency.length - 1]?.reason ?? "";
          const searchable = queries.length > 0;
          this.q.refinement = { searchable, reason, queries };
          this.q.plan.push({ round: it, source: "refinement", searchable, reason, queries });
          const st = this.lastActive("refine");
          if (st) st.loopmark = searchable;
          this.setSummary("refine", searchable ? "search again" : "stop");
        }
        break;
      }
      case "search":
        this.q.funnel.n_unique_urls = d.n_unique ?? this.q.funnel.n_unique_urls;
        this.q.funnel.n_queries = d.n_queries ?? this.q.funnel.n_queries;
        this.setSummary("search", `${d.n_unique ?? 0} unique URLs`);
        break;
      case "pick_links":
        this.setSummary("pick_links", (d.picked ?? []).length ? `${d.picked.length} URLs` : (d.note ?? ""));
        break;
      case "fetch": {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const pages = (d.pages ?? []).map((p: any) => ({ url: p.url, domain: domainOf(p.url), chars: p.chars ?? 0 }));
        this.q.pages = [...(this.q.pages ?? []), ...pages];
        this.q.funnel.n_pages_fetched = (this.q.pages ?? []).length;
        this.setSummary("fetch", d.n_ok != null ? `${d.n_ok}/${d.n_picked ?? d.n_ok} pages` : (d.note ?? ""));
        break;
      }
      case "chunk_and_index":
        this.q.funnel.n_chunks = d.n_chunks ?? this.q.funnel.n_chunks;
        this.setSummary("index", `${d.n_chunks ?? 0} chunks`);
        break;
      case "retrieve":
        this.setSummary("retrieve", d.top_cosine != null ? `top cos ${d.top_cosine}` : `${d.n_candidates ?? 0} cand`);
        break;
      case "rerank":
        this.setSummary("rerank", d.n != null ? `${d.n} rescored` : (d.note ?? ""));
        break;
      case "select_chunks": {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const chunks = (d.chunks ?? []).map((c: any) => ({
          id: c.id ?? "", domain: c.domain, url: c.url ?? "",
          rerank_score: c.rerank ?? null, retrieval_score: c.retrieval ?? null, text: c.text ?? "",
        }));
        // the full reranked pool (top-40) with kept/dropped flags — feeds the
        // 40 → 12 funnel in retrieve/rerank, identical to frozen run.json.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const candidates = (d.candidates ?? []).map((c: any) => ({
          id: c.id ?? "", domain: c.domain, url: c.url ?? "",
          retrieval_score: c.retrieval ?? null, rerank_score: c.rerank ?? null, selected: !!c.selected,
        }));
        const domains = d.domains ?? [];
        const sel: SelectionIter = {
          iteration: it, pool_size: candidates.length || null, n_selected: chunks.length,
          n_domains: domains.length, chunks, candidates,
        };
        this.q.selection = [...this.q.selection.filter((s) => s.iteration !== it), sel];
        this.setSummary("select", `${chunks.length} · ${domains.length} domains`);
        break;
      }
      case "sufficiency_check":
        // key by the 0-based iteration so StageDetail(judge, it) resolves it
        this.q.sufficiency = [
          ...this.q.sufficiency.filter((s) => s.iteration !== it),
          { iteration: it, sufficient: !!d.sufficient, reason: d.reason ?? "", n_chunks: d.n_chunks ?? 0 },
        ];
        this.setSummary("judge", d.sufficient ? "sufficient" : "insufficient");
        break;
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  searchQuery(d: any) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const results = (d.results ?? []).map((r: any) => ({
      title: r.title ?? "", url: r.url ?? "", domain: domainOf(r.url ?? ""), snippet: r.snippet ?? "",
    }));
    this.q.search.push({ round: this.iter, query: d.query ?? "", results });
    this.q.funnel.n_results += results.length;
  }

  answer(payload: AnswerPayload) {
    const a = payload.answer;
    const claims = (a as unknown as { claims?: unknown[] }).claims;
    this.q.answer = {
      disclaim: payload.disclaim, draft: "", answer_attempts: payload.answer_attempts,
      n_claims: claims ? claims.length : null, n_unsupported: payload.n_unsupported,
    };
    this.q.annotation = {
      text: a.text, placements: a.placements, citations: a.citations,
      n_citations: a.citations.length, n_markers: null,
    };
    this.q.disclaim = payload.disclaim;
    this.q.n_iterations = this.iter + 1;
    const ansSt = this.stations.find((s) => s.kind === "answer");
    if (ansSt) ansSt.summary = `${this.q.answer.n_claims ?? 0} claims`;
    const citeSt = this.stations.find((s) => s.kind === "cite");
    if (citeSt) citeSt.summary = `${a.citations.length} sources`;
  }
}
