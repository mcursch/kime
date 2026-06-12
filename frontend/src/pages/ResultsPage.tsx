import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import type { Attempt } from '../types/attempt';
import SkeletonOverlay from '../components/SkeletonOverlay';

/** Fetch a single attempt from the API. */
async function fetchAttempt(id: string): Promise<Attempt> {
  const res = await fetch(`/api/attempts/${id}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch attempt: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<Attempt>;
}

/** Badge colour keyed on technique_label. */
function scoreBadgeStyle(label: string): React.CSSProperties {
  const map: Record<string, string> = {
    Excellent: '#16a34a',
    Good: '#2563eb',
    'Needs Work': '#d97706',
    Poor: '#dc2626',
  };
  return {
    backgroundColor: map[label] ?? '#6b7280',
    color: '#fff',
    borderRadius: '0.375rem',
    padding: '0.2rem 0.65rem',
    fontSize: '0.85rem',
    fontWeight: 600,
    display: 'inline-block',
    marginLeft: '0.5rem',
  };
}

/** Camera-angle warning banner. Dismissible via the × button. */
function CameraWarningBanner({ onDismiss }: { onDismiss: () => void }) {
  return (
    <div
      role="alert"
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.75rem',
        backgroundColor: '#fef3c7',
        border: '1px solid #d97706',
        borderRadius: '0.5rem',
        padding: '0.75rem 1rem',
        marginBottom: '1.5rem',
      }}
    >
      <span style={{ fontSize: '1.2rem', lineHeight: 1 }}>⚠️</span>
      <span style={{ flex: 1, color: '#92400e', fontSize: '0.95rem' }}>
        <strong>Camera angle warning:</strong> The filming angle may cause
        inaccurate scoring. For best results, film from directly in front of or
        to the side of your body.
      </span>
      <button
        aria-label="Dismiss camera angle warning"
        onClick={onDismiss}
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          color: '#92400e',
          fontSize: '1.1rem',
          lineHeight: 1,
          padding: 0,
        }}
      >
        ×
      </button>
    </div>
  );
}

export default function ResultsPage() {
  const { id } = useParams<{ id: string }>();
  const [attempt, setAttempt] = useState<Attempt | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [warningDismissed, setWarningDismissed] = useState(false);

  useEffect(() => {
    if (!id) return;
    setWarningDismissed(false);
    setLoading(true);
    setError(null);
    fetchAttempt(id)
      .then((data) => {
        setAttempt(data);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : 'Unknown error');
      })
      .finally(() => {
        setLoading(false);
      });
  }, [id]);

  if (loading) {
    return (
      <main style={pageStyle}>
        <p style={{ color: '#6b7280' }}>Loading results…</p>
      </main>
    );
  }

  if (error || !attempt) {
    return (
      <main style={pageStyle}>
        <p style={{ color: '#dc2626' }}>
          {error ?? 'Attempt not found.'}
        </p>
      </main>
    );
  }

  const radarData = attempt.criterion_scores.map((c) => ({
    criterion: c.label,
    score: c.score,
  }));

  return (
    <main style={pageStyle}>
      {/* Camera-angle warning banner */}
      {!attempt.camera_angle_ok && !warningDismissed && (
        <CameraWarningBanner onDismiss={() => setWarningDismissed(true)} />
      )}

      {/* Header */}
      <h1 style={{ margin: '0 0 0.25rem', fontSize: '1.6rem' }}>
        {attempt.technique} Results
      </h1>
      <p style={{ color: '#6b7280', margin: '0 0 1.5rem', fontSize: '0.9rem' }}>
        {new Date(attempt.created_at).toLocaleString()}
      </p>

      {/* Overall score */}
      <section
        aria-label="Overall score"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          marginBottom: '2rem',
        }}
      >
        <div
          style={{
            width: 96,
            height: 96,
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #1d4ed8, #7c3aed)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          <span
            style={{ color: '#fff', fontSize: '2rem', fontWeight: 700 }}
          >
            {Math.round(attempt.overall_score)}
          </span>
        </div>
        <div>
          <div style={{ fontSize: '0.85rem', color: '#6b7280' }}>
            Overall score
          </div>
          <div style={{ fontSize: '1.1rem', fontWeight: 600, marginTop: '0.2rem' }}>
            {attempt.technique}
            <span style={scoreBadgeStyle(attempt.technique_label)}>
              {attempt.technique_label}
            </span>
          </div>
        </div>
      </section>

      {/* Video playback with skeleton overlay */}
      {attempt.video_url && (
        <section aria-label="Video playback" style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.1rem', marginBottom: '0.75rem' }}>
            Video Playback
          </h2>
          <SkeletonOverlay
            videoUrl={attempt.video_url}
            landmarks={attempt.landmark_frames}
            fps={attempt.landmark_fps}
          />
        </section>
      )}

      {/* Radar chart */}
      <section aria-label="Per-criterion radar chart" style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.1rem', marginBottom: '0.75rem' }}>
          Criterion Breakdown
        </h2>
        <ResponsiveContainer width="100%" height={360}>
          <RadarChart data={radarData} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
            <PolarGrid />
            <PolarAngleAxis dataKey="criterion" tick={{ fontSize: 13 }} />
            <PolarRadiusAxis
              angle={90}
              domain={[0, 100]}
              tick={{ fontSize: 11 }}
            />
            <Radar
              name="Score"
              dataKey="score"
              stroke="#4f46e5"
              fill="#4f46e5"
              fillOpacity={0.35}
            />
            <Tooltip
              formatter={(value: number) => [`${value}`, 'Score']}
            />
          </RadarChart>
        </ResponsiveContainer>
      </section>

      {/* Coaching feedback */}
      <section aria-label="Coaching feedback" style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.1rem', marginBottom: '0.75rem' }}>
          Coaching Feedback
        </h2>
        <div
          style={{
            background: '#f8fafc',
            border: '1px solid #e2e8f0',
            borderRadius: '0.5rem',
            padding: '1rem 1.25rem',
            lineHeight: 1.7,
            color: '#1e293b',
            whiteSpace: 'pre-wrap',
          }}
        >
          {attempt.coaching_feedback}
        </div>
      </section>
    </main>
  );
}

const pageStyle: React.CSSProperties = {
  maxWidth: 720,
  margin: '0 auto',
  padding: '2rem 1.5rem',
  fontFamily:
    "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
};
