import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { fetchAttempts } from "../api/attempts";
import type { Attempt, TechniqueType } from "../types/attempt";

// Human-readable labels for each technique type
const TECHNIQUE_LABELS: Record<TechniqueType, string> = {
  front_kick: "Front Kick",
  roundhouse_kick: "Roundhouse Kick",
  straight_punch: "Straight Punch",
};

// Distinct colours for each series in the chart
const TECHNIQUE_COLORS: Record<TechniqueType, string> = {
  front_kick: "#4f86f7",
  roundhouse_kick: "#f7864f",
  straight_punch: "#4ff79a",
};

const ALL_TECHNIQUES: TechniqueType[] = [
  "front_kick",
  "roundhouse_kick",
  "straight_punch",
];

// Each data point in the chart keyed by ISO date string
type ChartPoint = {
  date: string; // formatted date label
  front_kick?: number;
  roundhouse_kick?: number;
  straight_punch?: number;
};

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/**
 * Build an array of chart data points sorted chronologically.
 * Each point represents a single attempt and carries only the score
 * for its own technique; other technique keys are absent so Recharts
 * renders gaps rather than zeros for missing readings.
 */
function buildChartData(attempts: Attempt[]): ChartPoint[] {
  return [...attempts]
    .sort(
      (a, b) =>
        new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    )
    .map((attempt) => ({
      date: formatDate(attempt.created_at),
      [attempt.technique_type]: attempt.overall_score,
    }));
}

export default function HistoryPage() {
  const navigate = useNavigate();
  const [attempts, setAttempts] = useState<Attempt[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchAttempts()
      .then((data) => {
        if (!cancelled) setAttempts(data);
      })
      .catch((err: unknown) => {
        if (!cancelled)
          setError(err instanceof Error ? err.message : "Unknown error");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <main className="history-page">
        <p className="history-status">Loading history…</p>
      </main>
    );
  }

  if (error) {
    return (
      <main className="history-page">
        <p className="history-status history-error">Error: {error}</p>
      </main>
    );
  }

  if (attempts.length === 0) {
    return (
      <main className="history-page">
        <h1 className="history-title">Progress History</h1>
        <p className="history-empty">
          No attempts recorded yet. Upload a video to get started!
        </p>
      </main>
    );
  }

  // Sort table newest-first
  const sortedAttempts = [...attempts].sort(
    (a, b) =>
      new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  );

  const chartData = buildChartData(attempts);

  // Only render series that appear in the data
  const presentTechniques = ALL_TECHNIQUES.filter((t) =>
    attempts.some((a) => a.technique_type === t)
  );

  return (
    <main className="history-page">
      <h1 className="history-title">Progress History</h1>

      {/* Trend chart */}
      <section className="history-chart" aria-label="Score trend over time">
        <h2 className="history-section-title">Score Over Time</h2>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart
            data={chartData}
            margin={{ top: 8, right: 24, left: 0, bottom: 8 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} unit="%" />
            <Tooltip formatter={(value) => [`${value}%`]} />
            <Legend />
            {presentTechniques.map((technique) => (
              <Line
                key={technique}
                type="monotone"
                dataKey={technique}
                name={TECHNIQUE_LABELS[technique]}
                stroke={TECHNIQUE_COLORS[technique]}
                strokeWidth={2}
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
                connectNulls={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </section>

      {/* Attempts table */}
      <section aria-label="Attempt history table">
        <h2 className="history-section-title">All Attempts</h2>
        <table className="history-table">
          <thead>
            <tr>
              <th scope="col">Date</th>
              <th scope="col">Technique</th>
              <th scope="col">Score</th>
            </tr>
          </thead>
          <tbody>
            {sortedAttempts.map((attempt) => (
              <tr
                key={attempt.id}
                className="history-row"
                onClick={() => navigate(`/${attempt.id}`)}
                role="link"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    navigate(`/${attempt.id}`);
                  }
                }}
                aria-label={`View results for ${TECHNIQUE_LABELS[attempt.technique_type]} on ${formatDate(attempt.created_at)}`}
              >
                <td>{formatDate(attempt.created_at)}</td>
                <td>{TECHNIQUE_LABELS[attempt.technique_type]}</td>
                <td>{attempt.overall_score}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </main>
  );
}
