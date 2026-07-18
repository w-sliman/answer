import { RUN, GROUP_ORDER, groupOf } from "../data";

export function Selector({ onPick }: { onPick: (qid: string) => void }) {
  return (
    <div className="qgroups">
      {GROUP_ORDER.map((label) => {
        const qs = RUN.questions.filter((q) => groupOf(q) === label);
        if (!qs.length) return null;
        return (
          <div className="grp" key={label}>
            <div className="gh">
              <span className="gl">{label}</span>
              <span className="gr" />
            </div>
            <div className="qlist">
              {qs.map((q, i) => {
                const loop = q.n_iterations > 1;
                return (
                  <button className="qbtn" key={q.qid} onClick={() => onPick(q.qid)}>
                    <span className="qi">{String(i + 1).padStart(2, "0")}</span>
                    <span className="qt">{q.question}</span>
                    {loop ? (
                      <span className="qtag loop">↻ loop</span>
                    ) : q.disclaim ? (
                      <span className="qtag">disclaim</span>
                    ) : null}
                    <span className="qgo">→</span>
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
