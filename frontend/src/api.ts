// Client for the ragproject backend API.
// The base URL is configurable so the same build works in dev and production.
const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface Source {
  text: string;
  score: number;
}

export interface QueryResult {
  answer: string;
  sources: Source[];
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`Request to ${path} failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export async function ingest(text: string, source?: string): Promise<string[]> {
  const data = await postJson<{ chunk_ids: string[] }>("/ingest", { text, source });
  return data.chunk_ids;
}

export function query(question: string, k = 5): Promise<QueryResult> {
  return postJson<QueryResult>("/query", { question, k });
}
