import type { Attempt } from "../types/attempt";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function fetchAttempts(): Promise<Attempt[]> {
  const response = await fetch(`${API_BASE}/attempts`);
  if (!response.ok) {
    throw new Error(`Failed to fetch attempts: ${response.statusText}`);
  }
  return response.json() as Promise<Attempt[]>;
}
