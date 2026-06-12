import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getAttemptResult, type AttemptResult } from '../api/client';
import SkeletonOverlay from '../components/SkeletonOverlay';

// ---------------------------------------------------------------------------
// Radar chart — pure SVG, no external charting library needed
// ---------------------------------------------------------------------------

/**
 * Render a radar chart for per-criterion scores.
 *
 * @param scores  Record mapping criterion name → 0-1 score value.
 */
function RadarChart({ scores }: { scores: Record<string, number> }) {
  const entries = Object.entries(scores);
  if (entries.length === 0) return null;

  const size = 300;
  const center = size / 2;
  const radius = 110;
  const n = entries.length;
  const angleStep = (2 * Math.PI) / n;

  /** Cartesian point on the radar for a given axis index and radial fraction. */
  const getPoint = (idx: number, r: number) => {
    const angle = angleStep * idx - Math.PI / 2;
    return { x: center + r * Math.cos(angle), y: center + r * Math.sin(angle) };
  };

  const rings = [0.25, 0.5, 0.75, 1.0];

  const ringPolygon = (fraction: number) =>
    Array.from({ length: n }, (_, i) => {
      const p = getPoint(i, radius * fraction);
      return `${p.x},${p.y}`;
    }).join(' ');

  const scorePolygon = entries
    .map(([, value], i) => {
      const pct = Math.min(Math.max(value, 0), 1);
      const p = getPoint(i, radius * pct);
      return `${p.x},${p.y}`;
    })
    .join(' ');

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      aria-label="Criterion radar chart"
      style={{ display: 'block', margin: '0 auto' }}
    >
      {/* Background grid rings */}
      {rings.map((r, ri) => (
        <polygon
          key={ri}
          points={ringPolygon(r)}
          fill="none"
          stroke="#e2e8f0"
          strokeWidth={1}
        />
      ))}

      {/* Axis spokes */}
      {Array.from({ length: n }, (_, i) => {
        const outer = getPoint(i, radius);
        return (
          <line
            key={i}
            x1={center}
            y1={center}
            x2={outer.x}
            y2={outer.y}
            stroke="#e2e8f0"
            strokeWidth={1}
          />
        );
      })}

      {/* Score area */}
      <polygon
        points={scorePolygon}
        fill="rgba(99,102,241,0.25)"
        stroke="rgb(99,102,241)"
        strokeWidth={2}
      />

      {/* Axis labels */}
      {entries.map(([name], i) => {
        const pt = getPoint(i, radius + 22);
        return (
          <text
            key={i}
            x={pt.x}
            y={pt.y}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={10}
            fill="#374151"
          >
            {name}
          </text>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default function ResultsPage() {
  const { attemptId } = useParams<{ attemptId: string }>();
  const [result, setResult] = useState<AttemptResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!attemptId) return;
    setLoading(true);
    setError(null);
    getAttemptResult(attemptId)
      .then(setResult)
      .catch((err: unknown) =>
        setError(err instanceof Error ? err.message : 'Failed to load result'),
      )
      .finally(() => setLoading(false));
  }, [attemptId]);

  if (loading) {
    return (
      <main>
        <h1>Results</h1>
        <p>Loading…</p>
      </main>
    );
  }

  if (error) {
    return (
      <main>
        <h1>Results</h1>
        <p role="alert">{error}</p>
      </main>
    );
  }

  if (!result) return null;

  const techniqueLabel = (result.technique ?? '').replace(/_/g, ' ');
  const scoreEntries = Object.entries(result.scores);

  return (
    <main>
      <h1>Results</h1>

      {result.camera_angle_ok === false && (
        <section aria-label="Camera angle warning" role="alert">
          <p style={{ color: '#b45309', background: '#fef3c7', padding: '0.75rem', borderRadius: '0.375rem' }}>
            ⚠️ <strong>Poor camera angle detected.</strong> The subject appears
            to have been filmed from the side rather than the front. Scores may
            be unreliable — please re-record from a frontal or near-frontal
            angle for accurate feedback.
          </p>
        </section>
      )}

      <section aria-label="Overall score">
        <p>
          <strong>{techniqueLabel}</strong> — Overall score:{' '}
          <strong>{result.overall_score}</strong>
        </p>
        <p>
          <small>{new Date(result.created_at).toLocaleString()}</small>
        </p>
      </section>

      {scoreEntries.length > 0 && (
        <section aria-label="Criterion scores">
          <h2>Score Breakdown</h2>
          <RadarChart scores={result.scores} />
          <ul>
            {scoreEntries.map(([name, score]) => (
              <li key={name}>
                <strong>{name}</strong>: {Math.round(score * 100)}
                {result.criteria?.[name] !== undefined && (
                  <> (Δ {result.criteria[name].toFixed(2)})</>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}

      {result.video_url && (
        <section aria-label="Video playback">
          <h2>Video</h2>
          <SkeletonOverlay videoUrl={result.video_url} />
        </section>
      )}

      {result.feedback && (
        <section aria-label="Coaching feedback">
          <h2>Coaching Feedback</h2>
          <p>{result.feedback}</p>
        </section>
      )}
    </main>
  );
}
