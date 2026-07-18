import { useMemo, useState, type ReactNode } from "react";
import type { Annotation } from "../data";

const domainOf = (url: string) => {
  try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return url; }
};

// Normalize a chunk of prose for matching a per-sentence placement against a
// rendered paragraph/list-item: drop citation markers and markdown punctuation,
// collapse whitespace. Lets a placement's sentence be found inside a joined
// paragraph regardless of bold/bullet/marker noise on either side.
const normalize = (s: string) =>
  s.replace(/\[\d+\]/g, " ").replace(/[*_#`>]/g, " ").replace(/\s+/g, " ").trim().toLowerCase();

const BULLET = /^\s*[*-]\s+/;
const HEADING = /^\s*(#{1,6})\s+/;

// inline markdown -> React nodes, with `[n]` rendered as an interactive citation
const INLINE = /(\*\*([^*]+?)\*\*)|(\*([^*\n]+?)\*)|(_([^_\n]+?)_)|(\[(\d+)\])/g;
function inline(
  text: string, kp: string, onCite: (n: number) => void, openN: number | null,
): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0, i = 0, m: RegExpExecArray | null;
  INLINE.lastIndex = 0;
  while ((m = INLINE.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1]) out.push(<strong key={kp + "b" + i}>{m[2]}</strong>);
    else if (m[3]) out.push(<em key={kp + "i" + i}>{m[4]}</em>);
    else if (m[5]) out.push(<em key={kp + "u" + i}>{m[6]}</em>);
    else if (m[7]) {
      const n = Number(m[8]);
      out.push(
        <button key={kp + "c" + i} className="cite" aria-expanded={openN === n}
          onClick={() => onCite(n)}>[{n}]</button>,
      );
    }
    last = INLINE.lastIndex; i++;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

type Block =
  | { kind: "h"; level: number; text: string }
  | { kind: "p"; text: string }
  | { kind: "ul"; items: string[] };

// Group the markdown into blocks by BLANK lines (a single newline is a soft wrap
// that should flow, not break): paragraphs join their lines, headings and bullet
// lists become their own blocks.
function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  let para: string[] = [];
  const flush = () => { if (para.length) { blocks.push({ kind: "p", text: para.join(" ") }); para = []; } };

  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) { flush(); continue; }
    const h = HEADING.exec(line);
    if (h) { flush(); blocks.push({ kind: "h", level: h[1].length, text: line.replace(HEADING, "") }); continue; }
    if (BULLET.test(line)) {
      flush();
      const item = line.replace(BULLET, "");
      const prev = blocks[blocks.length - 1];
      if (prev && prev.kind === "ul") prev.items.push(item);
      else blocks.push({ kind: "ul", items: [item] });
      continue;
    }
    para.push(line);
  }
  flush();
  return blocks;
}

export function Answer({ annotation }: { annotation: Annotation }) {
  const [open, setOpen] = useState<string | null>(null); // "blockKey:n"
  const citeDomain = useMemo(() => {
    const map = new Map<number, string>();
    annotation.citations.forEach((c) => map.set(c.n, domainOf(c.url)));
    return map;
  }, [annotation]);

  const blocks = useMemo(() => parseBlocks(annotation.text), [annotation]);

  const onCite = (key: string, n: number) => {
    const id = `${key}:${n}`;
    setOpen((cur) => (cur === id ? null : id));
  };
  const openNfor = (key: string) =>
    open && open.startsWith(key + ":") ? Number(open.slice(key.length + 1)) : null;

  // The evidence popover for a clicked `[n]` inside a chunk: gather the quotes
  // from every placement whose sentence falls in this chunk and backs source n.
  const evidence = (key: string, chunk: string): ReactNode => {
    const n = openNfor(key);
    if (n == null) return null;
    const hay = normalize(chunk);
    const quotes: string[] = [];
    for (const p of annotation.placements) {
      const needle = normalize(p.line);
      if (!needle || !hay.includes(needle)) continue;
      for (const e of p.evidence) {
        if (e.n !== n) continue;
        for (const q of e.quotes) if (!quotes.includes(q)) quotes.push(q);
      }
    }
    if (!quotes.length) return null;
    return (
      <div className="pop">
        <div className="ph">■ [{n}] {citeDomain.get(n)} — supports this</div>
        {quotes.map((q, i) => <div className="qt" key={i}>“{q}”</div>)}
      </div>
    );
  };

  return (
    <div className="answer">
      {blocks.map((b, bi) => {
        const key = `b${bi}`;
        if (b.kind === "h") {
          const cls = `ahd${b.level <= 2 ? " lg" : ""}`;
          return <p key={key} className={cls}>{inline(b.text, key, () => {}, null)}</p>;
        }
        if (b.kind === "ul") {
          return (
            <ul key={key}>
              {b.items.map((it, ii) => {
                const ik = `${key}i${ii}`;
                return (
                  <li key={ik}>
                    {inline(it, ik, (n) => onCite(ik, n), openNfor(ik))}
                    {evidence(ik, it)}
                  </li>
                );
              })}
            </ul>
          );
        }
        return (
          <div key={key}>
            <p>{inline(b.text, key, (n) => onCite(key, n), openNfor(key))}</p>
            {evidence(key, b.text)}
          </div>
        );
      })}
    </div>
  );
}
