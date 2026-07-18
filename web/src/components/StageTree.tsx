import { useEffect, useRef } from "react";
import type { Question, StageKind } from "../data";
import { StageDetail } from "./StageDetail";

/** One rendered pipeline station — the unit both replay and live feed to the tree. */
export interface Station {
  key: string;
  kind: StageKind;
  iteration: number;
  label: string;
  state: "active" | "done";
  summary: string;
  loopmark?: boolean;
}

/** The shared pipeline spine: identical stations + `StageDetail` for both the
 *  replay (frozen `run.json`) and the live (SSE-accumulated) views. */
export function StageTree({ stations, q, open, onToggle, follow = true }: {
  stations: Station[];
  q: Question;
  open: Set<string>;
  onToggle: (key: string) => void;
  follow?: boolean;
}) {
  const activeRef = useRef<HTMLElement | null>(null);
  const activeKey = stations.find((s) => s.state === "active")?.key ?? null;

  // follow the running step into view as playback / streaming advances
  useEffect(() => {
    if (!follow || !activeKey) return;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    activeRef.current?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "nearest" });
  }, [activeKey, follow]);

  return (
    <div className="flow">
      {stations.map((s) => {
        const isOpen = open.has(s.key);
        return (
          <section
            className={`stn ${isOpen ? "open" : ""} ${s.loopmark ? "loopmark" : ""}`}
            data-s={s.state}
            data-iter={s.iteration}
            ref={s.key === activeKey ? activeRef : undefined}
            key={s.key}
          >
            {s.loopmark && (
              <div className="loopback" aria-hidden="true">
                <span className="lt">↻ insufficient — searching again</span>
              </div>
            )}
            <div className="node" />
            <button className="shead" onClick={() => onToggle(s.key)}>
              <h2>{s.label}</h2>
              {s.iteration > 0 && <span className="it">iter {s.iteration}</span>}
              <span className="sm">{s.summary}</span>
              <span className="cx">{isOpen ? "hide" : "expand"} ›</span>
            </button>
            {isOpen && (
              <div className="sbody">
                <StageDetail kind={s.kind} iteration={s.iteration} q={q} />
              </div>
            )}
          </section>
        );
      })}
    </div>
  );
}
