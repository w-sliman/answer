import { useEffect, useMemo, useState } from "react";
import type { Event, Question } from "../data";
import { buildTimeline } from "../data";
import { usePlayback } from "../playback";
import { StageTree, type Station } from "./StageTree";
import { Answer } from "./Answer";

const domainOf = (url: string) => {
  try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return url; }
};

function summary(e: Event, q: Question): string {
  switch (e.kind) {
    case "plan": return `${q.plan.find((r) => r.round === 0)?.queries.length ?? 0} queries`;
    case "search": {
      const rs = q.search.filter((s) => s.round === e.iteration);
      return `${rs.reduce((a, s) => a + s.results.length, 0)} results`;
    }
    case "fetch": return `${q.funnel.n_pages_fetched} pages`;
    case "index": return `${q.funnel.n_chunks} chunks`;
    case "pick_links": return `${q.pages?.length ?? 0} sources`;
    case "retrieve": {
      const sel = q.selection.find((s) => s.iteration === e.iteration);
      const top = Math.max(0, ...(sel?.chunks.map((c) => c.retrieval_score ?? 0) ?? [0]));
      return sel ? `top cos ${top.toFixed(2)}` : "";
    }
    case "rerank": {
      const sel = q.selection.find((s) => s.iteration === e.iteration);
      return sel ? `${sel.chunks.length} rescored` : "";
    }
    case "select": {
      const sel = q.selection.find((s) => s.iteration === e.iteration);
      return `top ${sel?.n_selected ?? 12}${sel?.pool_size ? ` of ${sel.pool_size}` : ""}`;
    }
    case "judge": {
      const s = q.sufficiency.find((x) => x.iteration === e.iteration);
      return s ? (s.sufficient ? "sufficient" : "insufficient") : "";
    }
    case "refine": return q.refinement?.searchable ? "search again" : "stop";
    case "answer": return `${q.answer.n_claims ?? 0} claims`;
    case "cite": return `${q.annotation.n_citations ?? 0} sources`;
    default: return "";
  }
}

export function Player({ question, onBack }: { question: Question; onBack: () => void }) {
  const events = useMemo(() => buildTimeline(question), [question]);
  const pb = usePlayback(events);
  const [open, setOpen] = useState<Set<string>>(new Set());
  const [showSteps, setShowSteps] = useState(true);

  // reset per question
  useEffect(() => { setOpen(new Set()); setShowSteps(true); }, [events]);

  // Steps play in view; once the answer lands, collapse the tree so the answer
  // sits right under the question (and re-open the steps on replay).
  useEffect(() => {
    if (pb.phase === "running") setShowSteps(true);
    else if (pb.phase === "done") setShowSteps(false);
  }, [pb.phase]);

  const toggle = (id: string) =>
    setOpen((prev) => {
      const n = new Set(prev);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });

  const f = question.funnel;
  const done = pb.phase === "done";
  const statusText =
    pb.phase === "idle"
      ? "Ready — press Ask to replay the run."
      : pb.phase === "running"
        ? pb.currentStatus
        : question.answer.disclaim
          ? "Done — the sources don't answer this, so it disclaims honestly."
          : "Done — answered and cited from the sources.";

  // the stations that have started, in order — the shared tree renders these
  const stations: Station[] = [];
  events.forEach((e, i) => {
    const state = pb.stageState(i);
    if (state === "pending") return;
    stations.push({
      key: e.id, kind: e.kind, iteration: e.iteration, label: e.label,
      state, summary: summary(e, question),
      loopmark: e.kind === "refine" && (question.refinement?.searchable ?? false),
    });
  });

  return (
    <section className="run">
      <button className="back" onClick={onBack}>← all questions</button>
      <h1 className="qhead">{question.question}</h1>

      {/* live status line — updates in place, with movement */}
      <div className="console">
        {pb.phase === "running" && <span className="spin" />}
        <span className={`stext ${done ? "done" : ""}`} key={`${pb.phase}:${pb.index}`}>
          {statusText}
        </span>
        <div className="ctrls">
          {pb.phase === "idle" ? (
            <button className="btn primary" onClick={pb.start}>Ask ▸</button>
          ) : (
            <>
              <button
                className={`btn${done && !showSteps ? " flash" : ""}`}
                onClick={() => setShowSteps((s) => !s)}
              >
                {showSteps ? "Hide steps" : "Show all steps"}
              </button>
              {pb.phase === "running" && <button className="btn" onClick={pb.skip}>skip</button>}
              {done && <button className="btn" onClick={pb.replay}>↻ replay</button>}
            </>
          )}
        </div>
      </div>
      <div className="pbar"><i style={{ width: `${pb.progress * 100}%` }} /></div>

      {/* the pipeline steps — the shared spine, each clearly expandable */}
      {showSteps && (
        <StageTree stations={stations} q={question} open={open} onToggle={toggle} />
      )}

      {/* the ANSWER, printed under the whole tree */}
      {done && (
        <div className="result">
          <div className="rlabel">
            {question.answer.disclaim ? "Answer · honest disclaim" : "Answer"}
            <span className="rmeta">
              {question.answer.n_claims ?? 0} claims · {question.answer.n_unsupported ?? 0} unsupported
              · {question.annotation.n_citations ?? 0} sources
            </span>
          </div>
          <div className="card">
            <Answer annotation={question.annotation} />
          </div>
          <div className="card srcs">
            {question.annotation.citations.map((c) => (
              <div className="src" key={c.n}>
                <span className="mk">[{c.n}]</span>
                <span className="u">{domainOf(c.url)}</span>
                <span className="qn">{c.quotes.length} quote{c.quotes.length !== 1 ? "s" : ""}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* the retrieval metric, under the answer */}
      {done && (
        <div className="funnel" aria-label="retrieval funnel">
          {[
            [f.n_queries, "queries"],
            [f.n_results, "results"],
            [f.n_pages_fetched, "pages"],
            [f.n_chunks, "chunks"],
            [question.selection[0]?.n_selected ?? 12, "selected"],
          ].map(([n, cap], i, arr) => (
            <div key={cap} style={{ display: "contents" }}>
              <div className={`fstep ${i === arr.length - 1 ? "hi" : ""}`}>
                <div className="fnum">{(n as number).toLocaleString()}</div>
                <div className="fcap">{cap}</div>
              </div>
              {i < arr.length - 1 && <div className="farrow">→</div>}
            </div>
          ))}
        </div>
      )}

      <footer>
        <b>frozen run</b> · {question.qid} · replayed offline — no live search, no model calls
      </footer>
    </section>
  );
}
