import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  uploadVideo,
  getJobStatus,
  type Technique,
  type JobStatus,
} from '../api/client';

// ── Local state types ────────────────────────────────────────────────────────

type RecordState = 'idle' | 'requesting' | 'recording';
type SubmitState = 'idle' | 'uploading' | 'polling' | 'failed';

const TECHNIQUES: { value: Technique; label: string }[] = [
  { value: 'front_kick', label: 'Front Kick' },
  { value: 'roundhouse_kick', label: 'Roundhouse Kick' },
  { value: 'straight_punch', label: 'Straight Punch' },
];

const POLL_INTERVAL_MS = 2_000;
const MAX_POLLS = 150; // ~5 minutes

// ── Component ────────────────────────────────────────────────────────────────

export default function UploadPage() {
  const navigate = useNavigate();

  // Technique selection
  const [technique, setTechnique] = useState<Technique>('front_kick');

  // File picker
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [pickedFile, setPickedFile] = useState<File | null>(null);

  // Camera recording
  const [recordState, setRecordState] = useState<RecordState>('idle');
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const liveVideoRef = useRef<HTMLVideoElement>(null);
  const playbackVideoRef = useRef<HTMLVideoElement>(null);

  // Submission & polling
  const [submitState, setSubmitState] = useState<SubmitState>('idle');
  const [pollStatus, setPollStatus] = useState<JobStatus | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Derived: whichever video the user has prepared
  const videoToSubmit: File | null =
    pickedFile ??
    (recordedBlob
      ? new File([recordedBlob], 'recording.webm', { type: recordedBlob.type })
      : null);

  const isProcessing = submitState === 'uploading' || submitState === 'polling';

  // ── Camera helpers ─────────────────────────────────────────────────────────

  const startRecording = async () => {
    setErrorMessage(null);
    setRecordState('requesting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      streamRef.current = stream;

      if (liveVideoRef.current) {
        liveVideoRef.current.srcObject = stream;
      }

      const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9'
        : 'video/webm';

      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        setRecordedBlob(blob);
        // Stop live camera feed
        stream.getTracks().forEach((t) => t.stop());
        if (liveVideoRef.current) {
          liveVideoRef.current.srcObject = null;
        }
        if (playbackVideoRef.current) {
          playbackVideoRef.current.src = URL.createObjectURL(blob);
        }
        setRecordState('idle');
      };

      recorder.start();
      setRecordState('recording');
    } catch (err) {
      setRecordState('idle');
      setErrorMessage(
        `Camera error: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
  };

  // ── File picker handler ────────────────────────────────────────────────────

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    setPickedFile(file);
    // Clear any previous recording when a file is explicitly chosen
    setRecordedBlob(null);
  };

  // ── Polling loop ───────────────────────────────────────────────────────────

  const pollUntilDone = async (jobId: string, attemptId: string) => {
    for (let i = 0; i < MAX_POLLS; i++) {
      await new Promise<void>((r) => setTimeout(r, POLL_INTERVAL_MS));

      let status;
      try {
        status = await getJobStatus(jobId);
      } catch {
        // Transient network error — keep polling
        continue;
      }

      setPollStatus(status.status);

      if (status.status === 'completed') {
        navigate(`/${attemptId}`);
        return;
      }

      if (status.status === 'failed') {
        setSubmitState('failed');
        setErrorMessage(
          status.error ?? 'Analysis failed. Please try a different video.',
        );
        return;
      }
    }

    // Poll timeout
    setSubmitState('failed');
    setErrorMessage('Analysis is taking too long. Please try again later.');
  };

  // ── Form submission ────────────────────────────────────────────────────────

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!videoToSubmit) return;

    setSubmitState('uploading');
    setErrorMessage(null);
    setPollStatus(null);

    try {
      const { job_id, attempt_id } = await uploadVideo(videoToSubmit, technique);
      setSubmitState('polling');
      setPollStatus('pending');
      await pollUntilDone(job_id, attempt_id);
    } catch (err) {
      setSubmitState('failed');
      setErrorMessage(
        `Upload failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  };

  const dismissError = () => {
    setSubmitState('idle');
    setErrorMessage(null);
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <main>
      <h1>Upload</h1>
      <p>Upload or record a video of your technique to receive scored feedback.</p>

      <form onSubmit={handleSubmit} noValidate>
        {/* Technique selector */}
        <div>
          <label htmlFor="technique">Technique</label>
          <select
            id="technique"
            value={technique}
            onChange={(e) => setTechnique(e.target.value as Technique)}
            disabled={isProcessing}
          >
            {TECHNIQUES.map(({ value, label }) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </div>

        {/* ── File picker ── */}
        <fieldset disabled={isProcessing}>
          <legend>Choose a video file</legend>
          <input
            id="video-file"
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileChange}
          />
          {pickedFile && (
            <p aria-live="polite">Selected: {pickedFile.name}</p>
          )}
        </fieldset>

        {/* ── Camera record ── */}
        <fieldset disabled={isProcessing}>
          <legend>— or record with your camera —</legend>

          {/* Live preview while recording */}
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <video
            ref={liveVideoRef}
            autoPlay
            muted
            playsInline
            aria-label="Live camera preview"
            style={{
              display: recordState === 'recording' ? 'block' : 'none',
              maxWidth: '100%',
              maxHeight: '300px',
            }}
          />

          {/* Playback after recording */}
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <video
            ref={playbackVideoRef}
            controls
            playsInline
            aria-label="Recorded video preview"
            style={{
              display: recordedBlob && recordState === 'idle' ? 'block' : 'none',
              maxWidth: '100%',
              maxHeight: '300px',
            }}
          />

          {recordState === 'idle' && (
            <button type="button" onClick={startRecording}>
              {recordedBlob ? 'Re-record' : 'Start Recording'}
            </button>
          )}

          {recordState === 'requesting' && (
            <span aria-live="polite">Requesting camera access…</span>
          )}

          {recordState === 'recording' && (
            <button type="button" onClick={stopRecording}>
              Stop Recording
            </button>
          )}

          {recordedBlob && recordState === 'idle' && (
            <p aria-live="polite">
              Recording ready ({Math.round(recordedBlob.size / 1024)} KB)
            </p>
          )}
        </fieldset>

        {/* Submit button */}
        <button type="submit" disabled={!videoToSubmit || isProcessing}>
          {submitState === 'uploading'
            ? 'Uploading…'
            : submitState === 'polling'
              ? `Analysing… (${pollStatus ?? 'pending'})`
              : 'Analyse'}
        </button>
      </form>

      {/* Progress indicator */}
      {isProcessing && (
        <div role="status" aria-live="polite" aria-atomic="true">
          {submitState === 'uploading'
            ? 'Uploading video to server…'
            : `Analysing technique — status: ${pollStatus ?? 'pending'}…`}
        </div>
      )}

      {/* Error message */}
      {errorMessage && (
        <div role="alert">
          <p>{errorMessage}</p>
          <button type="button" onClick={dismissError}>
            Dismiss
          </button>
        </div>
      )}
    </main>
  );
}
