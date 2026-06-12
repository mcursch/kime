import { useParams } from 'react-router-dom';

export default function ResultsPage() {
  const { attemptId } = useParams<{ attemptId: string }>();

  return (
    <main>
      <h1>Results</h1>
      <p>Showing results for attempt: <code>{attemptId}</code></p>
    </main>
  );
}
