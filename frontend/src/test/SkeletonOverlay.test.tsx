import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import SkeletonOverlay, { type FrameLandmarks } from '../components/SkeletonOverlay';

// jsdom does not implement HTMLMediaElement playback APIs.
// Stub the ones the component touches so event listeners can be attached
// without errors.
beforeEach(() => {
  window.HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
  window.HTMLMediaElement.prototype.pause = vi.fn();
  // requestAnimationFrame is not available in jsdom — provide a no-op stub.
  vi.stubGlobal('requestAnimationFrame', vi.fn(() => 0));
  vi.stubGlobal('cancelAnimationFrame', vi.fn());
});

/** Build a minimal valid 33-landmark frame. */
function makeLandmarks(frameCount: number): FrameLandmarks[] {
  const singleFrame: FrameLandmarks = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.5,
    z: 0,
    visibility: 1,
  }));
  return Array.from({ length: frameCount }, () => singleFrame);
}

describe('SkeletonOverlay', () => {
  it('renders without crashing when landmarks is undefined', () => {
    const { container } = render(<SkeletonOverlay videoUrl="/test-video.mp4" />);
    expect(container.querySelector('video')).not.toBeNull();
    expect(container.querySelector('canvas')).not.toBeNull();
  });

  it('renders without crashing when a valid landmarks array is provided', () => {
    const landmarks = makeLandmarks(10);
    const { container } = render(
      <SkeletonOverlay videoUrl="/test-video.mp4" landmarks={landmarks} fps={30} />,
    );
    expect(container.querySelector('video')).not.toBeNull();
    expect(container.querySelector('canvas')).not.toBeNull();
  });
});
