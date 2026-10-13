import re
from typing import List, Dict, Any, Optional


class CitationParser:
    """
    Parses and verifies scientific paper citations from LLM generated responses,
    mapping each citation back to the grounded source chunk.
    """

    @staticmethod
    def extract_citations(
        text: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extracts citations from text (e.g., [Title, Section, p.X] or [Paper A, p.5])
        and maps them to chunks in retrieved_chunks.
        """
        citations = []
        seen_keys = set()

        # Match text within square brackets
        bracket_matches = re.finditer(r"\[(.*?)\]", text)

        for match in bracket_matches:
            raw_citation = match.group(1).strip()
            # Ignore standard markdown links or simple formatting
            if raw_citation.startswith("http") or raw_citation.startswith("Image"):
                continue

            # Try to match with retrieved chunks
            matched_chunk = CitationParser._match_chunk_to_citation(raw_citation, retrieved_chunks)
            if matched_chunk:
                key = (matched_chunk.get("paper_id"), matched_chunk.get("page_start"), matched_chunk.get("section"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    citations.append({
                        "paper_id": matched_chunk.get("paper_id", ""),
                        "paper_title": matched_chunk.get("paper_title", ""),
                        "section": matched_chunk.get("section"),
                        "page": matched_chunk.get("page_start"),
                        "chunk_text": matched_chunk.get("text", ""),
                        "relevance_score": matched_chunk.get("rerank_score") or matched_chunk.get("rrf_score")
                    })

        # If no explicit brackets matched but chunks were used, ground top retrieved chunks
        if not citations and retrieved_chunks:
            for c in retrieved_chunks[:2]:
                citations.append({
                    "paper_id": c.get("paper_id", ""),
                    "paper_title": c.get("paper_title", ""),
                    "section": c.get("section"),
                    "page": c.get("page_start"),
                    "chunk_text": c.get("text", ""),
                    "relevance_score": c.get("rerank_score") or c.get("rrf_score")
                })

        return citations

    @staticmethod
    def _match_chunk_to_citation(raw_citation: str, chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Fuzzy matches citation text against available chunks by title, page number, or section.
        """
        cit_lower = raw_citation.lower()

        # 1. Match by paper title substring
        for chunk in chunks:
            title = (chunk.get("paper_title") or "").lower()
            if title and (title in cit_lower or any(word in cit_lower for word in title.split() if len(word) > 4)):
                return chunk

        # 2. Match by page number (e.g. p.5, page 5)
        page_match = re.search(r"\b(?:p\.|page)\s*(\d+)\b", cit_lower)
        if page_match:
            target_page = int(page_match.group(1))
            for chunk in chunks:
                if chunk.get("page_start") == target_page or chunk.get("page_end") == target_page:
                    return chunk

        # 3. Match by section name
        for chunk in chunks:
            sec = (chunk.get("section") or "").lower()
            if sec and sec in cit_lower:
                return chunk

        # 4. Fallback: Check if citation refers to "Paper 1", "Paper A", "[1]"
        num_match = re.search(r"\b(?:paper\s*)?(\d+)\b", cit_lower)
        if num_match:
            idx = int(num_match.group(1)) - 1
            if 0 <= idx < len(chunks):
                return chunks[idx]

        return None
