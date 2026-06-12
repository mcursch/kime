import { useParams, Link } from "react-router-dom";

/**
 * ResultsPage – displays the detailed scoring breakdown for a single attempt.
 * This is a stub; full implementation is tracked in a separate issue.
 */
export default function ResultsPage() {
  const { attemptId } = useParams<{ attemptId: string }>();

  return (
    <main style={{ padding: "2rem" }}>
      <h1>Results</h1>
      <p>
        Showing results for attempt: <strong>{attemptId}</strong>
      </p>
      <Link to="/history">← Back to History</Link>
    </main>
  );
}
