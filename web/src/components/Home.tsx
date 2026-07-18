import { RUN } from "../data";
import { STAGE_ORDER, STAGE_META } from "../live";
import { Selector } from "./Selector";

const REPO = "https://github.com/w-sliman/answer";
const COLAB = "https://colab.research.google.com/github/w-sliman/answer/blob/main/demo/colab_live_demo.ipynb";
export function Home({ onPickReplay, onAskLive, backendOnline, base }: {
  onPickReplay: (qid: string) => void;
  onAskLive: () => void;
  backendOnline: boolean | null;
  base: string;
}) {
  const m = RUN.meta;
  return (
    <div className="home">
      <section className="pick hero">
        <div className="eyebrow"><span className="dot" /><span className="lbl">Agentic web search · RAG · LangGraph</span></div>
        <h1>Watch an agentic search engine <em>think.</em></h1>
        <p className="lede">
          <b>Answer</b> plans search queries, reads live web pages, ranks the evidence, and judges whether
          it’s enough — <b>looping back to search</b> when it isn’t — then writes a grounded answer with
          citations you can click, or <b>honestly disclaims</b> when the web doesn’t have it.
        </p>
        <div className="cta">
          <a className="btn primary" href="#demo">Try the demo ↓</a>
          <a className="btn" href={REPO} target="_blank" rel="noreferrer">GitHub ↗</a>
          <a className="btn" href={COLAB} target="_blank" rel="noreferrer">Run live on Colab ↗</a>
        </div>
      </section>

      <section className="demo" id="demo">
        <div className="grp"><div className="gh"><span className="gl">Try it</span><span className="gr" /></div></div>

        <div className="modes">
          <div className="mode on">
            <div className="mh">Replay a saved run</div>
            <p className="mp">
              Play back one real, frozen run — every stage, the data it produced, and clickable
              citations. Nothing live: it reads from a captured run below.
            </p>
          </div>
          <div className={`mode ${backendOnline ? "on" : "off"}`}>
            <div className="mh">
              Ask a new question
              <span className={`dotlive ${backendOnline ? "on" : ""}`} />
            </div>
            <p className="mp">
              {backendOnline
                ? "The backend is live — ask anything and watch the real pipeline run, streamed stage by stage."
                : backendOnline === null
                  ? "Checking for a live backend…"
                  : "No live backend detected. Launch the Colab notebook, open its URL, and reload here to enable live mode."}
            </p>
            <button className="btn primary" onClick={onAskLive} disabled={!backendOnline}>
              {backendOnline ? "Ask live ▸" : "Live offline"}
            </button>
          </div>
        </div>

        <Selector onPick={onPickReplay} />

        <p className="footnote">
          Replaying <b>{m.n_questions}</b> questions over <b>{m.n_pages_total}</b> pages and{" "}
          <b>{m.n_chunks_total.toLocaleString()}</b> chunks · model <span className="mono">{m.chat_model}</span>
        </p>
      </section>

      <section className="how">
        <div className="grp"><div className="gh"><span className="gl">How it works</span><span className="gr" /></div></div>
        <div className="pipe">
          {STAGE_ORDER.map((s, i) => (
            <span className="pchip" key={s}>
              {STAGE_META[s].label}{i < STAGE_ORDER.length - 1 ? <b>→</b> : null}
            </span>
          ))}
        </div>
        <div className="princ">
          <div className="pc"><h3>Citations you can’t fake</h3><p>The model cites integer chunk references, never URLs. Python resolves them deterministically — so a quote can’t point at the wrong page.</p></div>
          <div className="pc"><h3>Draft, then cite</h3><p>It writes freeform; a grounding pass binds each claim to its source; then a separate, model-free node places every <span className="mono">[n]</span>, each carrying its sentence’s exact quote.</p></div>
          <div className="pc"><h3>Loops when unsure</h3><p>An LLM judges whether the evidence is sufficient. If it isn’t — and budget remains — the planner takes a different angle and searches again.</p></div>
          <div className="pc"><h3>Disclaims honestly</h3><p>When the web doesn’t answer the question, it says so — and still cites the adjacent facts it did find, rather than guessing.</p></div>
        </div>
      </section>

      <section className="decision">
        <div className="grp"><div className="gh"><span className="gl">A decision worth showing</span><span className="gr" /></div></div>
        <div className="callout">
          <div className="ct">Gemma 4 · E2B → E4B</div>
          <p>
            The system first ran on <b>Gemma 4 E2B</b> (~2B active params). One question kept under-answering:
            the sources gave both a headline figure and the real, model-specific maximum, and E2B returned only
            the headline. <b>No prompt variant fixed it</b> — across many A/Bs, none beat the baseline. It was a
            model-capacity limit, not a wording bug.
          </p>
          <p>
            Swapping to <b>Gemma 4 E4B</b> (config-only, no code change) cracked it: the same retrieved chunks now
            yield the full, correct answer. The lesson kept: <b>know when you’re fighting the prompt vs. the model</b>,
            and let a measured A/B tell you which.
          </p>
        </div>
      </section>

      <footer className="homefoot">
        <a href={REPO} target="_blank" rel="noreferrer">GitHub</a><span>·</span>
        <a href={COLAB} target="_blank" rel="noreferrer">Colab</a><span>·</span>
        <span className="mono">{base ? `backend: ${base}` : "backend: same origin"}</span>
      </footer>
    </div>
  );
}
