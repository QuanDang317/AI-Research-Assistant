import { Paper, ChatMessage, ComparisonRow } from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function fetchPapers(): Promise<Paper[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/papers`);
  if (!res.ok) throw new Error('Failed to fetch papers');
  return res.json();
}

export async function uploadPaper(file: File): Promise<Paper> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE_URL}/api/v1/papers/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deletePaper(paperId: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/v1/papers/${paperId}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Failed to delete paper');
}

export async function sendChatMessage(params: {
  conversationId?: string;
  query: string;
  paperIds?: string[];
  topK?: number;
}): Promise<{ conversation_id: string; answer: string; citations: any[] }> {
  const res = await fetch(`${API_BASE_URL}/api/v1/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      conversation_id: params.conversationId,
      query: params.query,
      paper_ids: params.paperIds,
      top_k: params.topK || 5,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function comparePapers(paperIds: string[]): Promise<{
  summary: string;
  table: ComparisonRow[];
}> {
  const res = await fetch(`${API_BASE_URL}/api/v1/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paper_ids: paperIds }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
