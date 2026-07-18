import { useEffect, useReducer, useRef, useState } from "react";
import { askLive, STAGE_META, STAGE_ORDER, type AnswerPayload } from "../live";
import { LiveRun } from "../liveRun";
import { StageTree } from "./StageTree";
import { Answer } from "./Answer";

const domainOf = (url: string) => {
  try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return url; }
};
const cap = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s);

export function LivePlayer({ base, onBack }: { base: string; onBack: () => void }) {
  const [q, setQ] = useState("");
  const [phase, setPhase] = useState<"idle" | "running" | "done" | "error">("idle");
  const [status, setStatus] = useState("Ask a question — it runs live against the backend.");
  const [answer, setAnswer] = useState<AnswerPayload | null>(null);
  const [err, setErr] = useState("");
  const [open, setOpen] = useState<Set<string>>(new Set());
  const runRef = useRef<LiveRun>(new LiveRun(""));
  const [, force] = useReducer((n: number) => n + 1, 0);
  const abortRef = useRef<AbortController | null>(null);

  const run = runRef.current;
  const running = phase === "running";
  const done = phase === "done";
  const doneCount = run.stations.filter((s) => s.state === "done").length;
  const progress = done ? 1 : Math.min(0.95, doneCount / STAGE_ORDER.length);

  // settle the status line once the run finishes (it otherwise keeps the last stage)
  useEffect(() => {
    if (phase === "done") {
      setStatus(answer?.disclaim
        ? "Done — the sources don’t answer this, so it disclaims honestly."
        : "Done — answered and cited from the sources.");
    }
  }, [phase, answer]);

  const toggle = (key: string) =>
    setOpen((prev) => { const n = new Set(prev); n.has(key) ? n.delete(key) : n.add(key); return n; });

  async function ask() {
    const question = q.trim();
    if (!question || running) return;
    runRef.current = new LiveRun(question);
    setOpen(new Set()); setAnswer(null); setErr(""); setPhase("running");
    setStatus("Starting the run…"); force();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const r = runRef.current;
    try {
      await askLive(base, question, {
        onStageStart: (stage) => { r.stageStart(stage); setStatus(cap(STAGE_META[stage]?.desc ?? stage) + "…"); force(); },
        onStageData: (stage, data) => { r.stageData(stage, data); force(); },
        onSearchQuery: (data) => { r.searchQuery(data); force(); },
        onStageEnd: (stage) => { r.stageEnd(stage); force(); },
        onAnswer: (payload) => { r.answer(payload); setAnswer(payload); force(); },
        onError: (m) => { setErr(m); setPhase("error"); },
        onDone: () => setPhase((p) => (p === "error" ? p : "done")),
      }, ctrl.signal);
      setPhase((p) => (p === "error" ? p : "done"));
    } catch (e: any) {
      if (ctrl.signal.aborted) return;
      setErr(e?.message ?? String(e)); setPhase("error");
    }
  }

  function cancel() { abortRef.current?.abort(); setPhase("idle"); setStatus("Cancelled."); }

  const showTree = running || done || run.stations.length > 0;

  return (
    <section className="run">
      <button className="back" onClick={onBack}>← back</button>
      <h1 className="qhead">Ask a live question</h1>

      <div className="asker">
        <textarea
          className="askbox"
          placeholder="e.g. How does LangGraph differ from LangChain?"
          value={q}
          rows={2}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask(); }}
        />
        <div className="askctrls">
          <span className="askhint">⌘/Ctrl + ↵</span>
          {running
            ? <button className="btn" onClick={cancel}>stop</button>
            : <button className="btn primary" onClick={ask} disabled={!q.trim()}>Ask ▸</button>}
        </div>
      </div>

      {err && <div className="card errcard"><b>Couldn’t reach the pipeline.</b> {err}</div>}

      {showTree && (
        <>
          <div className="console">
            {running && <span className="spin" />}
            <span className={`stext ${done ? "done" : ""}`} key={status}>{status}</span>
          </div>
          <div className="pbar"><i style={{ width: `${progress * 100}%` }} /></div>
          <StageTree stations={run.stations} q={run.q} open={open} onToggle={toggle} />
        </>
      )}

      {answer && (
        <div className="result">
          <div className="rlabel">
            {answer.disclaim ? "Answer · honest disclaim" : "Answer"}
            <span className="rmeta">
              {run.q.answer.n_claims != null ? `${run.q.answer.n_claims} claims · ` : ""}
              {answer.n_unsupported != null ? `${answer.n_unsupported} unsupported · ` : ""}
              {answer.answer.citations.length} source{answer.answer.citations.length !== 1 ? "s" : ""}
            </span>
          </div>
          <div className="card"><Answer annotation={run.q.annotation} /></div>
          {answer.answer.citations.length > 0 && (
            <div className="card srcs">
              {answer.answer.citations.map((c) => (
                <div className="src" key={c.n}>
                  <span className="mk">[{c.n}]</span>
                  <span className="u">{domainOf(c.url)}</span>
                  <span className="qn">{c.quotes.length} quote{c.quotes.length !== 1 ? "s" : ""}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <footer><b>live</b> · streamed from the backend at {base || "this origin"} · no replay</footer>
    </section>
  );
}
