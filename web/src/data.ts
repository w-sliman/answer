import runJson from "./run.json";

// --- shapes of run.json (produced by eval/build_site_data.py) ---------------

export interface SearchResult { title: string; url: string; domain: string; snippet: string; }
export interface Query { query: string; why?: string; }
export interface PlanRound { round: number; source: string; searchable: boolean; reason: string; queries: Query[]; }
export interface SearchRound { round: number; query: string; results: SearchResult[]; }
export interface Funnel { n_queries: number; n_results: number; n_unique_urls: number; n_pages_fetched: number; n_chunks: number; }
export interface PageInfo { url: string; domain: string; chars: number; }
export interface ChunkPreview { domain: string; url: string; text: string; }
export interface Chunk { id: string; domain: string; url: string; rerank_score: number | null; retrieval_score: number | null; text: string; }
// One entry of the full retrieve/rerank pool (top-40), with both scores and
// whether MMR kept it — this is what drives the 40 → 12 funnel view.
export interface Candidate { id: string; domain: string; url: string; retrieval_score: number | null; rerank_score: number | null; selected: boolean; }
export interface SelectionIter { iteration: number; pool_size: number | null; n_selected: number; n_domains: number | null; chunks: Chunk[]; candidates: Candidate[]; }
export interface SuffIter { iteration: number; sufficient: boolean; reason: string; n_chunks: number; }
export interface Refinement { searchable: boolean; reason: string; queries: Query[]; }
export interface Citation { n: number; url: string; quotes: string[]; }
export interface PlacementSrc { n: number; url: string; quotes: string[]; }
export interface Placement { line: string; markers: number[]; evidence: PlacementSrc[]; }
export interface AnswerData {
  disclaim: boolean; draft: string; answer_attempts: number | null;
  n_claims: number | null; n_unsupported: number | null;
}
export interface Annotation { text: string; placements: Placement[]; citations: Citation[]; n_citations: number | null; n_markers: number | null; }

export interface Question {
  qid: string; question: string; disclaim: boolean; n_iterations: number;
  plan: PlanRound[]; search: SearchRound[]; funnel: Funnel;
  pages?: PageInfo[]; index_sample?: ChunkPreview[];
  selection: SelectionIter[]; sufficiency: SuffIter[]; refinement: Refinement | null;
  answer: AnswerData; annotation: Annotation;
}

export interface RunMeta {
  dataset: string; generated: string; chat_model: string; embed_model: string;
  embed_dim: number; search_backend: string; chunk_size: number; chunk_overlap: number;
  n_pages_total: number; n_chunks_total: number; n_questions: number;
}

export interface Run { meta: RunMeta; questions: Question[]; }

export const RUN = runJson as unknown as Run;

// --- playback timeline ------------------------------------------------------

// The real pipeline nodes, shown granularly (retrieve/rerank/select are the
// differentiating RAG stages — never merged).
export type StageKind =
  | "plan" | "search" | "pick_links" | "fetch" | "index"
  | "retrieve" | "rerank" | "select" | "judge" | "refine" | "answer" | "cite";

export interface Event {
  id: string;
  kind: StageKind;
  iteration: number;
  label: string;   // node label on the spine
  status: string;  // "now doing X" line during playback
  ms: number;      // compressed playback duration
}

const DUR: Record<StageKind, number> = {
  plan: 2400, search: 3600, pick_links: 2200, fetch: 3000, index: 2400,
  retrieve: 2200, rerank: 2600, select: 2600, judge: 3000, refine: 2800,
  answer: 3400, cite: 2400,
};

function ev(kind: StageKind, iteration: number, label: string, status: string, seq: number): Event {
  return { id: `${kind}-${iteration}-${seq}`, kind, iteration, label, status, ms: DUR[kind] };
}

/** Compile one question into the ordered playback events, expanding the
 *  refinement loop into a real second pass when it happened. */
export function buildTimeline(q: Question): Event[] {
  const out: Event[] = [];
  let seq = 0;
  const push = (k: StageKind, it: number, label: string, status: string) =>
    out.push(ev(k, it, label, status, seq++));

  const nq0 = q.plan.find((r) => r.round === 0)?.queries.length ?? 0;
  const nRes = (round: number) =>
    q.search.filter((s) => s.round === round).reduce((a, s) => a + s.results.length, 0);
  const nSel = (it: number) => q.selection.find((s) => s.iteration === it)?.n_selected ?? 0;

  // retrieve → rerank → select is one sub-sequence per iteration
  const retrieval = (it: number, again: boolean) => {
    push("retrieve", it, "Retrieve",
      again ? "Re-embedding the query — dense top-K candidates…" : "Embedding the query — dense top-K candidates…");
    push("rerank", it, "Rerank", "Cross-encoder rescoring the candidates…");
    push("select", it, "Select", `MMR — ${nSel(it)} diverse passages…`);
  };

  // --- iteration 0 ---
  push("plan", 0, "Plan", `Planning the search — ${nq0} ${nq0 === 1 ? "query" : "queries"}…`);
  push("search", 0, "Search", `Searching the web — ${nRes(0)} results…`);
  push("pick_links", 0, "Pick sources", "Choosing which pages are worth reading…");
  push("fetch", 0, "Read", `Reading ${q.funnel.n_pages_fetched} pages…`);
  push("index", 0, "Index", `Chunking & embedding — ${q.funnel.n_chunks} passages…`);
  retrieval(0, false);
  const suff0 = q.sufficiency.find((s) => s.iteration === 0);
  push("judge", 0, "Judge", "Checking whether the sources are enough…");

  const insufficient0 = suff0 ? !suff0.sufficient : false;
  const ref = q.refinement;

  if (insufficient0 && ref) {
    if (!ref.searchable) {
      // STOP — unfillable by search
      push("refine", 1, "Refine", "Not enough — but no search can fill the gap. Stopping.");
    } else {
      // SEARCH — a real second pass
      push("refine", 1, "Refine", `Not enough — refining the search (${ref.queries.length} new queries)…`);
      push("search", 1, "Search", `Searching again — ${nRes(1)} new results…`);
      push("pick_links", 1, "Pick sources", "Choosing new pages to read…");
      push("fetch", 1, "Read", "Reading the new pages…");
      push("index", 1, "Index", "Indexing the new passages…");
      retrieval(1, true);
      push("judge", 1, "Judge", "Checking again whether it's enough…");
    }
  }

  // --- answer ---
  if (q.answer.disclaim) {
    push("answer", 0, "Answer", "Writing an honest answer — the sources don't cover this…");
  } else {
    push("answer", 0, "Answer", "Writing the answer from the sources…");
  }
  push("cite", 0, "Cite", "Grounding every claim & adding citations…");
  return out;
}

// Question grouping for the selector — derived from the ACTUAL frozen run
// (disclaim outcome + planned query count), NOT from qid/authoring assumptions.
// The dataset's `category` records design intent (stable); the demo groups by what
// the pipeline actually did, so the labels can never drift from the results. (E.g.
// the `refinement_rescue` questions answered in one pass here, and two
// `adversarial_with_context` questions turned out answerable — so both land in
// "Single-query answer", which is what actually happened.)
export const GROUP_ORDER = ["Multi-query planning", "Single-query answer", "Disclaimed"] as const;
export type GroupLabel = (typeof GROUP_ORDER)[number];

export function groupOf(q: Question): GroupLabel {
  if (q.disclaim) return "Disclaimed";
  const nq = q.plan.find((r) => r.round === 0)?.queries.length ?? 0;
  return nq >= 2 ? "Multi-query planning" : "Single-query answer";
}
