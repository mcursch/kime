import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import ResultsPage from './ResultsPage';
import * as client from '../api/client';

// ── Mocks ────────────────────────────────────────────────────────────────────

vi.mock('../api/client', () => ({
  getAttemptResult: vi.fn(),
}));

// SkeletonOverlay uses requestAnimationFrame; stub it for the test environment.
vi.stubGlobal('requestAnimationFrame', vi.fn(() => 0));
vi.stubGlobal('cancelAnimationFrame', vi.fn());

// ── Fixtures ─────────────────────────────────────────────────────────────────

const MOCK_RESULT: client.AttemptResult = {
  attempt_id: 'attempt-42',
  technique: 'front_kick',
  overall_score: 78,
  criteria: [
    { name: 'Chamber Height', score: 8, max_score: 10, delta_from_reference: -2, feedback: 'Good lift.' },
    { name: 'Hip Rotation', score: 7, max_score: 10, delta_from_reference: -3, feedback: 'Rotate more.' },
  ],
  coaching_feedback: 'Focus on your chamber.',
  video_url: '/video/attempt-42.mp4',
  created_at: '2026-01-01T12:00:00Z',
};

// ── Helper ───────────────────────────────────────────────────────────────────

function renderPage(attemptId = 'attempt-42') {
  return render(
    <MemoryRouter initialEntries={[`/${attemptId}`]}>
      <Routes>
        <Route path="/:attemptId" element={<ResultsPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('ResultsPage', () => {
  beforeEach(() => {
    vi.mocked(client.getAttemptResult).mockReset();
  });

  it('shows a loading state while fetching', () => {
    // Never resolves during this test
    vi.mocked(client.getAttemptResult).mockReturnValue(new Promise(() => {}));
    renderPage();
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it('renders the overall score and criteria after a successful fetch', async () => {
    vi.mocked(client.getAttemptResult).mockResolvedValue(MOCK_RESULT);
    renderPage();

    await waitFor(() => expect(screen.queryByText(/loading/i)).not.toBeInTheDocument());

    expect(screen.getByText(/overall score/i)).toBeInTheDocument();
    expect(screen.getByText(/78/)).toBeInTheDocument();
    expect(screen.getAllByText(/Chamber Height/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Hip Rotation/).length).toBeGreaterThan(0);
  });

  it('renders the coaching feedback', async () => {
    vi.mocked(client.getAttemptResult).mockResolvedValue(MOCK_RESULT);
    renderPage();

    await waitFor(() => expect(screen.queryByText(/loading/i)).not.toBeInTheDocument());

    expect(screen.getByText('Focus on your chamber.')).toBeInTheDocument();
  });

  it('renders the radar chart SVG', async () => {
    vi.mocked(client.getAttemptResult).mockResolvedValue(MOCK_RESULT);
    const { container } = renderPage();

    await waitFor(() => expect(screen.queryByText(/loading/i)).not.toBeInTheDocument());

    expect(container.querySelector('svg[aria-label="Criterion radar chart"]')).not.toBeNull();
  });

  it('renders a video element for skeleton overlay', async () => {
    vi.mocked(client.getAttemptResult).mockResolvedValue(MOCK_RESULT);
    const { container } = renderPage();

    await waitFor(() => expect(screen.queryByText(/loading/i)).not.toBeInTheDocument());

    expect(container.querySelector('video')).not.toBeNull();
  });

  it('shows an error alert when the fetch fails', async () => {
    vi.mocked(client.getAttemptResult).mockRejectedValue(new Error('Not found'));
    renderPage();

    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
    expect(screen.getByRole('alert')).toHaveTextContent('Not found');
  });
});
