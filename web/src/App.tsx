import { useEffect, useState } from "react";
import { RUN } from "./data";
import { Home } from "./components/Home";
import { Player } from "./components/Player";
import { LivePlayer } from "./components/LivePlayer";
import { getApiBase, pingBackend } from "./live";

type View = { kind: "home" } | { kind: "run"; qid: string } | { kind: "live" };

export function App() {
  const [view, setView] = useState<View>({ kind: "home" });
  const [base] = useState<string>(() => getApiBase());
  const [online, setOnline] = useState<boolean | null>(null);

  // Detect a live backend once (same origin, or a ?api= override).
  useEffect(() => {
    let alive = true;
    pingBackend(base).then((ok) => { if (alive) setOnline(ok); });
    return () => { alive = false; };
  }, [base]);

  const toggleTheme = () => {
    const root = document.documentElement;
    root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
  };

  const goHome = () => setView({ kind: "home" });
  const question = view.kind === "run" ? RUN.questions.find((q) => q.qid === view.qid) ?? null : null;

  return (
    <>
      <header>
        <div className="bar">
          <button className="brand" onClick={goHome}>
            ANSWER<b>/</b>{view.kind === "live" ? "live" : "replay"}
          </button>
          <span className="sep" />
          <span className={`livechip ${online ? "on" : online === false ? "off" : ""}`}>
            {online == null ? "checking…" : online ? "● live backend" : "○ replay only"}
          </span>
          <button className="tog" onClick={toggleTheme} aria-label="Toggle colour theme">theme</button>
        </div>
      </header>

      <main className="wrap">
        {view.kind === "run" && question ? (
          <Player question={question} onBack={goHome} />
        ) : view.kind === "live" ? (
          <LivePlayer base={base} onBack={goHome} />
        ) : (
          <Home
            onPickReplay={(qid) => setView({ kind: "run", qid })}
            onAskLive={() => setView({ kind: "live" })}
            backendOnline={online}
            base={base}
          />
        )}
      </main>
    </>
  );
}
