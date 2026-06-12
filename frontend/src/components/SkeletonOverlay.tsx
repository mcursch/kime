import { useEffect, useRef } from 'react';

/** A single MediaPipe pose landmark (normalized 0–1 image coordinates). */
export interface Landmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

/** All 33 landmarks for a single video frame. */
export type FrameLandmarks = Landmark[];

/**
 * MediaPipe Pose landmark connections (index pairs).
 * Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
 */
const POSE_CONNECTIONS: [number, number][] = [
  // Face
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  // Torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // Right arm
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  // Left arm
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  // Right leg
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  // Left leg
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
];

const LINE_COLOR = 'rgba(99, 102, 241, 0.85)';  // indigo
const DOT_COLOR = 'rgba(244, 63, 94, 0.9)';      // rose
const DOT_RADIUS = 4;
const LINE_WIDTH = 2;
/** Landmarks below this visibility score are skipped. */
const VISIBILITY_THRESHOLD = 0.5;

export interface SkeletonOverlayProps {
  /** URL of the video to play. */
  videoUrl: string;
  /**
   * Per-frame landmark arrays (index 0 = frame 0).
   * Omitting or passing null/undefined is a graceful no-op —
   * only the video plays without any skeleton overlay.
   */
  landmarks?: FrameLandmarks[] | null;
  /**
   * Frames per second of the landmark data, used to map
   * `video.currentTime` → frame index. Defaults to 30.
   */
  fps?: number;
}

/**
 * HTML5 video player with a `<canvas>` overlay that draws the MediaPipe
 * 33-landmark pose skeleton synchronized to playback time.
 *
 * Skeleton drawing uses `requestAnimationFrame` and does not block playback.
 * Scrubbing triggers an immediate `seeked` redraw so the overlay stays in sync.
 * When `landmarks` is absent or empty the component renders as a plain video
 * player with no errors.
 */
export default function SkeletonOverlay({
  videoUrl,
  landmarks,
  fps = 30,
}: SkeletonOverlayProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafIdRef = useRef<number | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    /**
     * Resize the canvas buffer to match the video element's rendered size,
     * then draw the skeleton frame corresponding to `currentTime`.
     */
    function drawFrame(currentTime: number) {
      if (!canvas || !video) return;

      // Sync canvas buffer dimensions to the displayed video size.
      const displayW = video.clientWidth || video.videoWidth;
      const displayH = video.clientHeight || video.videoHeight;
      if (displayW > 0 && displayH > 0) {
        if (canvas.width !== displayW) canvas.width = displayW;
        if (canvas.height !== displayH) canvas.height = displayH;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Graceful no-op when landmark data is absent.
      if (!landmarks || landmarks.length === 0) return;

      const frameIndex = Math.min(
        Math.floor(currentTime * fps),
        landmarks.length - 1,
      );
      const frame = landmarks[frameIndex];
      if (!frame || frame.length < 33) return;

      const { width, height } = canvas;

      // --- Skeleton lines ---
      ctx.strokeStyle = LINE_COLOR;
      ctx.lineWidth = LINE_WIDTH;
      ctx.lineCap = 'round';
      for (const [a, b] of POSE_CONNECTIONS) {
        const la = frame[a];
        const lb = frame[b];
        if (!la || !lb) continue;
        if (
          (la.visibility !== undefined && la.visibility < VISIBILITY_THRESHOLD) ||
          (lb.visibility !== undefined && lb.visibility < VISIBILITY_THRESHOLD)
        ) {
          continue;
        }
        ctx.beginPath();
        ctx.moveTo(la.x * width, la.y * height);
        ctx.lineTo(lb.x * width, lb.y * height);
        ctx.stroke();
      }

      // --- Joint dots ---
      ctx.fillStyle = DOT_COLOR;
      for (const lm of frame) {
        if (
          lm.visibility !== undefined &&
          lm.visibility < VISIBILITY_THRESHOLD
        ) {
          continue;
        }
        ctx.beginPath();
        ctx.arc(lm.x * width, lm.y * height, DOT_RADIUS, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    /** RAF loop — continuously redraws while the video is playing. */
    function loop() {
      drawFrame(video!.currentTime);
      rafIdRef.current = requestAnimationFrame(loop);
    }

    function startLoop() {
      if (rafIdRef.current !== null) return;
      rafIdRef.current = requestAnimationFrame(loop);
    }

    function stopLoop() {
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
    }

    /** Immediate redraw when the user scrubs. */
    function onSeeked() {
      drawFrame(video!.currentTime);
    }

    /** Size the canvas once the video dimensions are known. */
    function onLoadedMetadata() {
      drawFrame(video!.currentTime);
    }

    video.addEventListener('play', startLoop);
    video.addEventListener('pause', stopLoop);
    video.addEventListener('ended', stopLoop);
    video.addEventListener('seeked', onSeeked);
    video.addEventListener('loadedmetadata', onLoadedMetadata);

    // Draw the initial/paused frame.
    drawFrame(video.currentTime);

    // If the video is already playing when props change (causing effect re-run),
    // re-start the RAF loop — no 'play' event will fire in this case.
    if (!video.paused && !video.ended) {
      startLoop();
    }

    return () => {
      video.removeEventListener('play', startLoop);
      video.removeEventListener('pause', stopLoop);
      video.removeEventListener('ended', stopLoop);
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('loadedmetadata', onLoadedMetadata);
      stopLoop();
    };
  }, [landmarks, fps]);

  return (
    <div
      style={{
        position: 'relative',
        display: 'inline-block',
        width: '100%',
        lineHeight: 0,
      }}
    >
      <video
        ref={videoRef}
        src={videoUrl}
        controls
        playsInline
        style={{
          display: 'block',
          width: '100%',
          borderRadius: '0.5rem',
        }}
      />
      <canvas
        ref={canvasRef}
        aria-hidden="true"
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
          borderRadius: '0.5rem',
        }}
      />
    </div>
  );
}
