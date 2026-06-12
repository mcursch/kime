import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { listAttempts, type AttemptSummary } from '../api/client';

const PAGE_SIZE = 10;

// ---------------------------------------------------------------------------
// Trend chart — simple SVG line chart showing overall_score over time
// ---------------------------------------------------------------------------

function TrendChart({ attempts }: { attempts: AttemptSummary[] }) {
  if (attempts.length < 2) return null;

  const sorted = [...attempts].sort(
    (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
  );

  const width = 400;
  const height = 160;
  const padL = 40;
  const padR = 16;
  const padT = 16;
  const padB = 32;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const scores = sorted.map((a) => a.overall_score);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const scoreRange = maxScore - minScore || 1;

  const xOf = (i: number) =>
    padL + (i / (sorted.length - 1)) * plotW;
  const yOf = (s: number) =>
    padT + (1 - (s - minScore) / scoreRange) * plotH;

  const polyPoints = sorted
    .map((a, i) => `${xOf(i)},${yOf(a.overall_score)}`)
    .join(' ');

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      aria-label="Score trend chart"
      style={{ display: 'block' }}
    >
      {/* Axes */}
      <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="#e2e8f0" strokeWidth={1} />
      <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="#e2e8f0" strokeWidth={1} />

      {/* Score line */}
      <polyline points={polyPoints} fill="none" stroke="rgb(99,102,241)" strokeWidth={2} />

      {/* Data points */}
      {sorted.map((a, i) => (
        <circle
          key={a.attempt_id}
          cx={xOf(i)}
          cy={yOf(a.overall_score)}
          r={4}
          fill="rgb(99,102,241)"
        />
      ))}

      {/* Y-axis labels */}
      <text
        x={padL - 6}
        y={padT}
        textAnchor="end"
        dominantBaseline="middle"
        fontSize={10}
        fill="#6b7280"
      >
        {maxScore}
      </text>
      <text
        x={padL - 6}
        y={padT + plotH}
        textAnchor="end"
        dominantBaseline="middle"
        fontSize={10}
        fill="#6b7280"
      >
        {minScore}
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default function HistoryPage() {
  const navigate = useNavigate();
  const [attempts, setAttempts] = useState<AttemptSummary[]>([]);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    listAttempts(PAGE_SIZE, offset)
      .then((page) => {
        setAttempts((prev) => (offset === 0 ? page : [...prev, ...page]));
        setHasMore(page.length === PAGE_SIZE);
      })
      .catch((err: unknown) =>
        setError(err instanceof Error ? err.message : 'Failed to load history'),
      )
      .finally(() => setLoading(false));
  }, [offset]);

  return (
    <main>
      <h1>History</h1>

      {error && <p role="alert">{error}</p>}

      {attempts.length >= 2 && (
        <section aria-label="Score trend">
          <h2>Score Trend</h2>
          <TrendChart attempts={attempts} />
        </section>
      )}

      {!loading && attempts.length === 0 && !error && (
        <p>No attempts yet.</p>
      )}

      {attempts.length > 0 && (
        <ul>
          {attempts.map((a) => (
            <li key={a.attempt_id}>
              <button type="button" onClick={() => navigate(`/${a.attempt_id}`)}>
                {new Date(a.created_at).toLocaleDateString()} —{' '}
                {a.technique.replace(/_/g, ' ')} — Score: {a.overall_score}
              </button>
            </li>
          ))}
        </ul>
      )}

      {loading && <p>Loading…</p>}

      {!loading && hasMore && (
        <button type="button" onClick={() => setOffset((o) => o + PAGE_SIZE)}>
          Load more
        </button>
      )}
    </main>
  );
}
