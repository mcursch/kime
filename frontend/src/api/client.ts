/**
 * Typed API client for the Kime FastAPI backend.
 *
 * All functions return typed responses and throw on non-2xx HTTP status.
 * The base URL defaults to the Vite dev-proxy target and can be overridden
 * via the VITE_API_BASE_URL environment variable.
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '';

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export type Technique = 'front_kick' | 'roundhouse_kick' | 'straight_punch';

export type JobStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface UploadResponse {
  /** UUID that identifies the async analysis job. */
  job_id: string;
  /** UUID that will identify the attempt once analysis is complete. */
  attempt_id: string;
}

export interface JobStatusResponse {
  job_id: string;
  attempt_id: string;
  status: JobStatus;
  /** ISO-8601 timestamp when the job was created. */
  created_at: string;
  /** ISO-8601 timestamp when the job finished (null while pending/processing). */
  finished_at: string | null;
  error: string | null;
}

export interface CriterionScore {
  name: string;
  score: number;
  max_score: number;
  delta_from_reference: number;
  feedback: string;
}

export interface AttemptResult {
  attempt_id: string;
  technique: Technique;
  overall_score: number;
  criteria: CriterionScore[];
  coaching_feedback: string;
  video_url: string;
  created_at: string;
}

export interface AttemptSummary {
  attempt_id: string;
  technique: Technique;
  overall_score: number;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { Accept: 'application/json', ...init?.headers },
    ...init,
  });

  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new Error(`API ${response.status}: ${text}`);
  }

  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

/**
 * Upload a video file for analysis.
 *
 * @param file      The video file chosen by the user.
 * @param technique The technique being performed in the video.
 * @returns         The job and attempt IDs for the queued analysis job.
 */
export async function uploadVideo(
  file: File,
  technique: Technique,
): Promise<UploadResponse> {
  const body = new FormData();
  body.append('file', file);
  body.append('technique', technique);

  return request<UploadResponse>('/api/upload', { method: 'POST', body });
}

/**
 * Poll the status of an async analysis job.
 *
 * @param jobId    The job_id returned by `uploadVideo`.
 * @param options  Optional fetch options (e.g. an AbortSignal).
 * @returns        Current status plus timestamps and any error message.
 */
export async function getJobStatus(
  jobId: string,
  options?: { signal?: AbortSignal },
): Promise<JobStatusResponse> {
  return request<JobStatusResponse>(`/api/jobs/${encodeURIComponent(jobId)}`, {
    signal: options?.signal,
  });
}

/**
 * Fetch the full scored result for a completed attempt.
 *
 * @param attemptId  The attempt_id returned by `uploadVideo` (or from history).
 * @returns          Scores, per-criterion breakdown, coaching feedback, and video URL.
 */
export async function getAttemptResult(attemptId: string): Promise<AttemptResult> {
  return request<AttemptResult>(`/api/attempts/${encodeURIComponent(attemptId)}`);
}

/**
 * List all past attempts for the current user/session, newest first.
 *
 * @param limit   Maximum number of results to return (default: 20).
 * @param offset  Pagination offset (default: 0).
 * @returns       Array of attempt summaries.
 */
export async function listAttempts(
  limit = 20,
  offset = 0,
): Promise<AttemptSummary[]> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  return request<AttemptSummary[]>(`/api/attempts?${params.toString()}`);
}
