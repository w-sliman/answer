import { useCallback, useEffect, useRef, useState } from "react";
import type { Event } from "./data";

export type Phase = "idle" | "running" | "done";
export type StageState = "pending" | "active" | "done";

const reduce = () =>
  typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Drives the compiled event timeline as a compressed, staged playback. */
export function usePlayback(events: Event[]) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [index, setIndex] = useState(-1); // active event index
  const timer = useRef<number | null>(null);

  const clear = () => {
    if (timer.current) { clearTimeout(timer.current); timer.current = null; }
  };

  // reset whenever the timeline changes (question switch)
  useEffect(() => {
    clear();
    setPhase("idle");
    setIndex(-1);
  }, [events]);

  // advance the active event on its duration
  useEffect(() => {
    if (phase !== "running" || index < 0) return;
    if (index >= events.length) { setPhase("done"); return; }
    const ms = reduce() ? 0 : events[index].ms;
    timer.current = window.setTimeout(() => setIndex((i) => i + 1), ms);
    return clear;
  }, [phase, index, events]);

  const start = useCallback(() => {
    clear();
    if (reduce()) { setIndex(events.length); setPhase("done"); return; }
    setPhase("running");
    setIndex(0);
  }, [events]);

  const skip = useCallback(() => { clear(); setIndex(events.length); setPhase("done"); }, [events]);
  const replay = useCallback(() => { clear(); setPhase("running"); setIndex(0); }, []);

  const stageState = useCallback(
    (i: number): StageState => {
      if (phase === "done") return "done";
      if (phase === "idle") return "pending";
      if (i < index) return "done";
      if (i === index) return "active";
      return "pending";
    },
    [phase, index],
  );

  const progress = phase === "done" ? 1 : index < 0 ? 0 : index / events.length;
  const currentStatus =
    phase === "running" && index >= 0 && index < events.length ? events[index].status : "";

  return { phase, index, start, skip, replay, stageState, currentStatus, progress };
}
