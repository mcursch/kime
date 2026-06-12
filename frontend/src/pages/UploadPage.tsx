import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadVideo, getJobStatus, type Technique } from '../api/client';

const POLL_INTERVAL_MS = 2_000;

const TECHNIQUES: { value: Technique; label: string }[] = [
  { value: 'front_kick', label: 'Front Kick' },
  { value: 'roundhouse_kick', label: 'Roundhouse Kick' },
  { value: 'straight_punch', label: 'Straight Punch' },
];

export default function UploadPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [technique, setTechnique] = useState<Technique>('front_kick');
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;

    setError(null);
    setUploading(true);

    try {
      const { job_id } = await uploadVideo(file, technique);

      // Poll until the job is complete or fails
      const intervalId = setInterval(async () => {
        try {
          const status = await getJobStatus(job_id);
          if (status.status === 'completed') {
            clearInterval(intervalId);
            navigate(`/${job_id}`);
          } else if (status.status === 'failed') {
            clearInterval(intervalId);
            setError(status.error_message ?? 'Analysis failed');
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
        <div>
          <label htmlFor="technique">Technique</label>
          <select
            id="technique"
            value={technique}
            onChange={(e) => setTechnique(e.target.value as Technique)}
            disabled={uploading}
          >
            {TECHNIQUES.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="video-file">Video file</label>
          <input id="video-file" type="file" accept="video/*" ref={fileInputRef} />
        </div>
        <button type="submit" disabled={uploading}>
          {uploading ? 'Analysing…' : 'Analyse'}
        </button>
      </form>
      {error && <div role="alert">{error}</div>}
    </main>
  );
}
