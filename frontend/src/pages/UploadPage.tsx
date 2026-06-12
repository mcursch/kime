import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadVideo, getJobStatus } from '../api/client';

const POLL_INTERVAL_MS = 2_000;

export default function UploadPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;

    setError(null);
    setUploading(true);

    try {
      const { job_id, attempt_id } = await uploadVideo(file, 'front_kick');

      // Poll until the job is complete or fails
      const intervalId = setInterval(async () => {
        try {
          const status = await getJobStatus(job_id);
          if (status.status === 'completed') {
            clearInterval(intervalId);
            navigate(`/${attempt_id}`);
          } else if (status.status === 'failed') {
            clearInterval(intervalId);
            setError(status.error ?? 'Analysis failed');
            setUploading(false);
          }
        } catch {
          // Transient network errors are silently retried
        }
      }, POLL_INTERVAL_MS);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setUploading(false);
    }
  }

  return (
    <main>
      <h1>Upload</h1>
      <p>Upload a video of your technique to get scored feedback.</p>
      <form onSubmit={handleSubmit}>
        <input id="video-file" type="file" accept="video/*" ref={fileInputRef} />
        <button type="submit" disabled={uploading}>
          Analyse
        </button>
      </form>
      {error && <div role="alert">{error}</div>}
    </main>
  );
}
