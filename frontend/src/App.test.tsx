import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';

// Minimal stubs so page components render without real API calls
vi.mock('./api/client', () => ({
  uploadVideo: vi.fn(),
  getJobStatus: vi.fn(),
  getAttemptResult: vi.fn(() => new Promise(() => {})),
  listAttempts: vi.fn(() => new Promise(() => {})),
}));

// requestAnimationFrame is used by SkeletonOverlay
vi.stubGlobal('requestAnimationFrame', vi.fn(() => 0));
vi.stubGlobal('cancelAnimationFrame', vi.fn());

// Import App after mocks are registered
import App from './App';

describe('App routing', () => {
  it('renders UploadPage at /', () => {
    // jsdom default URL is http://localhost/
    render(<App />);
    // UploadPage renders an Analyse button
    expect(screen.getByRole('button', { name: /analyse/i })).toBeInTheDocument();
  });
});
