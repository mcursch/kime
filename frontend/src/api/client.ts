/**
 * Typed API client for the Kime FastAPI backend.
 *
 * All functions return typed responses and throw on non-2xx HTTP status.
 * The base URL defaults to the Vite dev-proxy target and can be overridden
 * via the VITE_API_BASE_URL environment variable.
 *
 * Endpoint mapping (backend has no /api prefix):
 *   uploadVideo       → POST  /upload
 *   getJobStatus      → GET   /jobs/{job_id}
 *   getAttemptResult  → GET   /jobs/{job_id}/results
 *   listAttempts      → GET   /history?session_id=&page=&page_size=
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '';

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export type Technique = 'front_kick' | 'roundhouse_kick' | 'straight_punch';

export type JobStatus = 'pending' | 'processing' | 'completed' | 'failed';

/** Response from POST /upload. */
export interface UploadResponse {
  /** Integer primary-key of the created analysis job. */
  job_id: number;
  status: string;
}

/** Response from GET /jobs/{job_id}. */
export interface JobStatusResponse {
  job_id: number;
  status: JobStatus;
  error_message: string | null;
}

/** Full analysis result from GET /jobs/{job_id}/results (AnalysisResultResponse). */
export interface AttemptResult {
  /** UUID string identifying the job. */
  job_id: string;
  /** "complete" for completed jobs, otherwise the job status value. */
  status: string;
  technique: string | null;
  session_id: string | null;
  /** Per-criterion scores in 0–1 range. Keys are criterion slugs. */
  scores: Record<string, number>;
  /** Raw DTW-alignment deltas from reference. Keys are criterion slugs. */
  metric_deltas: Record<string, number>;
  keyframe_paths: string[];
  overall_score: number | null;
  /** Coaching feedback from Claude; null if not yet generated. */
  feedback: string | null;
  /**
   * Alias for metric_deltas exposed by the backend for API consumers.
   * Values are numeric deltas (same keys as metric_deltas).
   */
  criteria: Record<string, number> | null;
  created_at: string;
  /** Optional video URL when the backend provides one. */
  video_url?: string;
}

/** Summary item from GET /history (HistoryItem). */
export interface AttemptSummary {
  /** UUID string identifying the job. */
  job_id: string;
  technique: string;
  status: JobStatus;
  overall_score: number | null;
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
 * Calls POST /upload.
 *
 * @param file      The video file chosen by the user.
 * @param technique The technique being performed in the video.
 * @returns         The integer job_id and initial status.
 */
export async function uploadVideo(
  file: File,
  technique: Technique,
): Promise<UploadResponse> {
  const body = new FormData();
  body.append('file', file);
  body.append('technique', technique);

  return request<UploadResponse>('/upload', { method: 'POST', body });
}

/**
 * Poll the status of an async analysis job.
 *
 * Calls GET /jobs/{job_id}.
 *
 * @param jobId  The job_id returned by `uploadVideo`.
 * @returns      Current status and any error message.
 */
export async function getJobStatus(jobId: number | string): Promise<JobStatusResponse> {
  return request<JobStatusResponse>(`/jobs/${encodeURIComponent(String(jobId))}`);
}

/**
 * Fetch the full scored result for a completed job.
 *
 * Calls GET /jobs/{job_id}/results where job_id is the UUID string column.
 *
 * @param jobId  The job identifier (integer or UUID string) from history or a completed upload.
 * @returns      Scores, per-criterion breakdown, coaching feedback.
 */
export async function getAttemptResult(jobId: string): Promise<AttemptResult> {
  return request<AttemptResult>(`/jobs/${encodeURIComponent(jobId)}/results`);
}

/**
 * List past analysis jobs for a session, newest first.
 *
 * Calls GET /history?session_id=&page=&page_size= and returns the items array.
 *
 * @param limit     Maximum number of results to return (default: 20).
 * @param offset    Pagination offset (default: 0); converted to 1-based page.
 * @param sessionId Session identifier (default: empty string).
 * @returns         Array of job summaries.
 */
export async function listAttempts(
  limit = 20,
  offset = 0,
  sessionId = '',
): Promise<AttemptSummary[]> {
  const page = offset === 0 ? 1 : Math.floor(offset / limit) + 1;
  const params = new URLSearchParams({
    session_id: sessionId,
    page: String(page),
    page_size: String(limit),
  });
  const result = await request<{ items: AttemptSummary[] }>(`/history?${params.toString()}`);
  return result.items;
}
