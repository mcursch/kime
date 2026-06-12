import { act, render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import UploadPage from './UploadPage';
import * as client from '../api/client';

// ── Mocks ────────────────────────────────────────────────────────────────────

vi.mock('../api/client', () => ({
  uploadVideo: vi.fn(),
  getJobStatus: vi.fn(),
}));

const mockNavigate = vi.fn();
vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>();
  return { ...actual, useNavigate: () => mockNavigate };
});

// ── Helpers ──────────────────────────────────────────────────────────────────

function renderPage() {
  return render(
    <MemoryRouter>
      <UploadPage />
    </MemoryRouter>,
  );
}

/** Attach a fake video file to the file input (id="video-file"). */
function pickFile(name = 'kick.mp4') {
  const file = new File(['data'], name, { type: 'video/mp4' });
  const input = document.querySelector<HTMLInputElement>('#video-file')!;
  fireEvent.change(input, { target: { files: [file] } });
  return file;
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('UploadPage', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    mockNavigate.mockReset();
    vi.mocked(client.uploadVideo).mockReset();
    vi.mocked(client.getJobStatus).mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // (a) Successful upload → polling → navigate
  it('navigates to the attempt page after a successful upload and completed job', async () => {
    vi.mocked(client.uploadVideo).mockResolvedValue({
      job_id: 'job-1',
      attempt_id: 'attempt-1',
    });
    vi.mocked(client.getJobStatus).mockResolvedValue({
      job_id: 'job-1',
      attempt_id: 'attempt-1',
      status: 'completed',
      created_at: '',
      finished_at: '',
      error: null,
    });

    renderPage();
    pickFile();

    // Click submit — uploadVideo mock resolves immediately (microtask queue)
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /analyse/i }));
    });

    // Advance past the first poll interval (triggers getJobStatus, then navigate)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });

    expect(mockNavigate).toHaveBeenCalledWith('/attempt-1');
  });

  // (b) Failed job → error message rendered
  it('shows an error message when the job fails', async () => {
    vi.mocked(client.uploadVideo).mockResolvedValue({
      job_id: 'job-2',
      attempt_id: 'attempt-2',
    });
    vi.mocked(client.getJobStatus).mockResolvedValue({
      job_id: 'job-2',
      attempt_id: 'attempt-2',
      status: 'failed',
      created_at: '',
      finished_at: '',
      error: 'Pose estimation failed',
    });

    renderPage();
    pickFile();

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /analyse/i }));
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('Pose estimation failed');
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  // (c) Network error during polling is retried, not surfaced to the user
  it('retries silently on a transient network error and eventually navigates', async () => {
    vi.mocked(client.uploadVideo).mockResolvedValue({
      job_id: 'job-3',
      attempt_id: 'attempt-3',
    });

    // First poll throws a network error; second poll returns completed.
    vi.mocked(client.getJobStatus)
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValue({
        job_id: 'job-3',
        attempt_id: 'attempt-3',
        status: 'completed',
        created_at: '',
        finished_at: '',
        error: null,
      });

    renderPage();
    pickFile();

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /analyse/i }));
    });

    // First poll interval — getJobStatus rejects, loop continues silently
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });

    // No error alert shown after a transient failure
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();

    // Second poll interval — returns completed, navigate called
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });

    expect(mockNavigate).toHaveBeenCalledWith('/attempt-3');
  });
});
