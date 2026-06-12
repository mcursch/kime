import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import HistoryPage from './HistoryPage';
import * as client from '../api/client';

// ── Mocks ────────────────────────────────────────────────────────────────────

vi.mock('../api/client', () => ({
  listAttempts: vi.fn(),
}));

const mockNavigate = vi.fn();
vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>();
  return { ...actual, useNavigate: () => mockNavigate };
});

// ── Fixtures ─────────────────────────────────────────────────────────────────

function makeAttempt(n: number): client.AttemptSummary {
  return {
    attempt_id: `attempt-${n}`,
    technique: 'front_kick',
    overall_score: 60 + n,
    created_at: `2026-01-${String(n).padStart(2, '0')}T12:00:00Z`,
  };
}

const PAGE_1 = Array.from({ length: 10 }, (_, i) => makeAttempt(i + 1));
const PAGE_2 = Array.from({ length: 5 }, (_, i) => makeAttempt(i + 11));

// ── Helper ───────────────────────────────────────────────────────────────────

function renderPage() {
  return render(
    <MemoryRouter>
      <HistoryPage />
    </MemoryRouter>,
  );
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('HistoryPage', () => {
  beforeEach(() => {
    vi.mocked(client.listAttempts).mockReset();
    mockNavigate.mockReset();
  });

  it('shows a loading indicator while fetching', () => {
    vi.mocked(client.listAttempts).mockReturnValue(new Promise(() => {}));
    renderPage();
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it('renders an attempt list after a successful fetch', async () => {
    vi.mocked(client.listAttempts).mockResolvedValue(PAGE_1);
    renderPage();

    await waitFor(() => expect(screen.queryByText(/loading/i)).not.toBeInTheDocument());

    expect(screen.getAllByRole('listitem').length).toBe(10);
    expect(screen.getAllByText(/front kick/i).length).toBeGreaterThan(0);
  });

  it('shows "No attempts yet" when the history is empty', async () => {
    vi.mocked(client.listAttempts).mockResolvedValue([]);
    renderPage();

    await waitFor(() => expect(screen.queryByText(/loading/i)).not.toBeInTheDocument());

    expect(screen.getByText(/no attempts yet/i)).toBeInTheDocument();
  });

  it('renders an error alert when the fetch fails', async () => {
    vi.mocked(client.listAttempts).mockRejectedValue(new Error('Server error'));
    renderPage();

    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
    expect(screen.getByRole('alert')).toHaveTextContent('Server error');
  });

  it('navigates to the attempt page when a history item is clicked', async () => {
    vi.mocked(client.listAttempts).mockResolvedValue(PAGE_1);
    renderPage();

    await waitFor(() => expect(screen.queryByText(/loading/i)).not.toBeInTheDocument());

    // Click the first listed attempt button
    const buttons = screen.getAllByRole('button');
    const historyButton = buttons.find((b) => b.textContent?.includes('front kick'));
    fireEvent.click(historyButton!);

    expect(mockNavigate).toHaveBeenCalledWith(expect.stringMatching(/^\/attempt-/));
  });

  it('loads the next page when "Load more" is clicked', async () => {
    vi.mocked(client.listAttempts)
      .mockResolvedValueOnce(PAGE_1)
      .mockResolvedValueOnce(PAGE_2);

    renderPage();

    await waitFor(() => expect(screen.queryByText(/loading/i)).not.toBeInTheDocument());

    const loadMore = screen.getByRole('button', { name: /load more/i });
    fireEvent.click(loadMore);

    await waitFor(() => expect(screen.queryByText(/loading/i)).not.toBeInTheDocument());

    expect(screen.getAllByRole('listitem').length).toBe(15);
    // "Load more" should be gone because PAGE_2 has fewer than 10 items
    expect(screen.queryByRole('button', { name: /load more/i })).not.toBeInTheDocument();
  });

  it('renders the trend chart SVG when there are at least two attempts', async () => {
    vi.mocked(client.listAttempts).mockResolvedValue(PAGE_1);
    const { container } = renderPage();

    await waitFor(() => expect(screen.queryByText(/loading/i)).not.toBeInTheDocument());

    expect(container.querySelector('svg[aria-label="Score trend chart"]')).not.toBeNull();
  });
});
