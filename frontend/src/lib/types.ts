export interface Paper {
  id: string;
  title: string;
  authors: string[];
  year?: number;
  venue?: string;
  abstract?: string;
  field?: string;
  file_path: string;
  total_pages: number;
  status: string;
  created_at: string;
}

export interface Chunk {
  id: string;
  paper_id: string;
  chunk_index: number;
  section?: string;
  page_start?: number;
  page_end?: number;
  text: string;
  token_count: number;
}

export interface Citation {
  paper_id: string;
  paper_title: string;
  section?: string;
  page?: number;
  chunk_text: string;
  relevance_score?: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  created_at?: string;
}

export interface ComparisonRow {
  paper_id: string;
  paper_title: string;
  method: string;
  dataset: string;
  metric: string;
  result: string;
  citations: string[];
}
