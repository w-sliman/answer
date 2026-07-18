// Live-backend client: talks to api.py (FastAPI + SSE) so the site can ask a
// NEW question and stream the pipeline's real stages, instead of replaying the
// frozen run.json. Same UI, different data source.
import type { Annotation } from "./data";

// --- shapes returned by api.py ---------------------------------------------

export interface LiveCitation { n: number; url: string; quotes: string[] }
export interface LivePlacement {
  line: string; markers: number[];
  evidence: { n: number; url: string; quotes: string[] }[];
}
export interface LiveAnswer {
  text: string;
  citations: LiveCitation[];
  placements: LivePlacement[];
}
export interface AnswerPayload {
  answer: LiveAnswer;
  disclaim: boolean;
  iteration: number | null;
  answer_attempts: number | null;
  n_unsupported: number | null;
}

/** Adapt the live answer payload into the `Annotation` the <Answer> component
 *  already knows how to render (same shape the frozen run.json uses). */
export function toAnnotation(a: LiveAnswer): Annotation {
  return {
    text: a.text,
    placements: a.placements,
    citations: a.citations,
    n_citations: a.citations.length,
    n_markers: null,
  };
}

// --- backend discovery ------------------------------------------------------

const LS_KEY = "answer_api";

/** Resolve the API base URL. Precedence: `?api=<url>` (persisted) → saved value
 *  → same origin (""), which is exactly right when api.py serves this site. */
export function getApiBase(): string {
  try {
    const q = new URL(window.location.href).searchParams.get("api");
    if (q) { const clean = q.replace(/\/+$/, ""); localStorage.setItem(LS_KEY, clean); return clean; }
    const saved = localStorage.getItem(LS_KEY);
    if (saved) return saved;
  } catch { /* SSR / blocked storage — fall through */ }
  return "";
}

export function setApiBase(v: string): string {
  const clean = v.trim().replace(/\/+$/, "");
  try { clean ? localStorage.setItem(LS_KEY, clean) : localStorage.removeItem(LS_KEY); } catch { /* ignore */ }
  return clean;
}

/** True if `${base}/health` answers {ok:true} within a short timeout. */
export async function pingBackend(base: string): Promise<boolean> {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 4000);
    const r = await fetch(base + "/health", { signal: ctrl.signal });
    clearTimeout(t);
    if (!r.ok) return false;
    const j = await r.json().catch(() => null);
    return !!(j && j.ok);
  } catch { return false; }
}

// --- pipeline stage vocabulary ---------------------------------------------

export const STAGE_ORDER = [
  "generate_queries", "search", "pick_links", "fetch", "chunk_and_index",
  "retrieve", "rerank", "select_chunks", "sufficiency_check", "answer", "cite",
] as const;

export const STAGE_META: Record<string, { label: string; desc: string }> = {
  generate_queries: { label: "Plan", desc: "planning search queries" },
  search: { label: "Search", desc: "searching the web" },
  pick_links: { label: "Pick sources", desc: "choosing which pages to read" },
  fetch: { label: "Read", desc: "fetching pages" },
  chunk_and_index: { label: "Index", desc: "chunking & embedding" },
  retrieve: { label: "Retrieve", desc: "dense top-K candidates" },
  rerank: { label: "Rerank", desc: "cross-encoder rescoring" },
  select_chunks: { label: "Select", desc: "MMR for source diversity" },
  sufficiency_check: { label: "Judge", desc: "is the evidence enough?" },
  answer: { label: "Answer", desc: "drafting & grounding" },
  cite: { label: "Cite", desc: "placing citations" },
};

export function stageLabel(stage: string): string {
  return STAGE_META[stage]?.label ?? stage;
}

/** One-line summary from a stage_data event, for the station spine. */
export function liveSummary(stage: string, d: Record<string, any>): string {
  switch (stage) {
    case "generate_queries": { const n = (d.queries ?? []).length; return n ? `${n} quer${n === 1 ? "y" : "ies"}` : ""; }
    case "search": return d.n_unique != null ? `${d.n_unique} unique URLs` : "";
    case "pick_links": return (d.picked ?? []).length ? `${d.picked.length} URLs` : (d.note ?? "");
    case "fetch": return d.n_ok != null ? `${d.n_ok}/${d.n_picked ?? d.n_ok} pages` : (d.note ?? "");
    case "chunk_and_index": return d.n_chunks != null ? `${d.n_chunks} chunks` : "";
    case "retrieve": return d.n_candidates != null ? `${d.n_candidates} candidates` : "";
    case "rerank": return d.n != null ? `${d.n} scored` : (d.note ?? "");
    case "select_chunks": { const c = (d.chunks ?? []).length; const dm = (d.domains ?? []).length; return c ? `${c} chunks · ${dm} domains` : ""; }
    case "sufficiency_check": return d.sufficient != null ? (d.sufficient ? "sufficient" : "insufficient") : "";
    case "answer": return d.words != null ? `${d.words} words` : "";
    default: return "";
  }
}

// --- the streaming call -----------------------------------------------------

export interface LiveHandlers {
  onStart?: (question: string) => void;
  onStageStart?: (stage: string) => void;
  onStageEnd?: (stage: string, elapsed?: number) => void;
  onStageData?: (stage: string, data: Record<string, any>) => void;
  onSearchQuery?: (data: Record<string, any>) => void;
  onStatus?: (line: string) => void;
  onAnswer?: (payload: AnswerPayload) => void;
  onDone?: () => void;
  onError?: (message: string) => void;
}

/** POST /ask and parse the Server-Sent Events stream frame by frame. */
export async function askLive(
  base: string, question: string, h: LiveHandlers, signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(base + "/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (res.status === 409) throw new Error("The backend is already handling a question — try again in a moment.");
  if (!res.ok) throw new Error(`Backend returned HTTP ${res.status}.`);
  if (!res.body) throw new Error("The backend returned no event stream.");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      dispatchFrame(buf.slice(0, idx), h);
      buf = buf.slice(idx + 2);
    }
  }
}

function dispatchFrame(frame: string, h: LiveHandlers): void {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    // lines beginning ":" are keep-alive comments — ignored
  }
  if (!dataLines.length) return;
  let data: any;
  try { data = JSON.parse(dataLines.join("\n")); } catch { return; }

  switch (event) {
    case "start": h.onStart?.(data.question ?? ""); break;
    case "event":
      if (data.kind === "stage_start") h.onStageStart?.(data.stage);
      else if (data.kind === "stage_end") h.onStageEnd?.(data.stage, data.elapsed);
      else if (data.kind === "stage_data") h.onStageData?.(data.stage, data);
      else if (data.kind === "search_query_end") h.onSearchQuery?.(data);
      break;
    case "status": h.onStatus?.(data.line ?? ""); break;
    case "answer": h.onAnswer?.(data as AnswerPayload); break;
    case "done": h.onDone?.(); break;
    case "error": h.onError?.(data.message ?? "Unknown backend error."); break;
  }
}
