import re
from typing import List, Dict, Any, Optional


class SemanticChunker:
    """
    Chunks academic paper sections into semantically cohesive passages
    with metadata, target token sizes (500-1000 tokens / approx 400-750 words),
    and configured sliding overlap.
    """

    def __init__(self, target_word_size: int = 500, overlap_word_size: int = 75):
        self.target_word_size = target_word_size
        self.overlap_word_size = overlap_word_size

    def chunk_paper(self, parsed_paper: Dict[str, Any], paper_id: str) -> List[Dict[str, Any]]:
        """
        Takes the output of PDFParser and chunks all sections.
        Returns a list of chunk dicts ready for embedding and database storage.
        """
        chunks = []
        chunk_idx = 0

        title = parsed_paper.get("title", "")
        authors = parsed_paper.get("authors", [])
        year = parsed_paper.get("year", None)
        sections = parsed_paper.get("sections", [])

        # If no structured sections were identified, treat pages as fallback
        if not sections and "pages" in parsed_paper:
            sections = [
                {
                    "section_title": f"Page {p['page']}",
                    "page_start": p["page"],
                    "page_end": p["page"],
                    "content": p["text"]
                }
                for p in parsed_paper["pages"] if p["text"].strip()
            ]

        for sec in sections:
            sec_title = sec.get("section_title", "General")
            sec_content = sec.get("content", "")
            page_start = sec.get("page_start", 1)
            page_end = sec.get("page_end", page_start)

            if not sec_content.strip():
                continue

            # Split section content into paragraphs
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", sec_content) if p.strip()]

            current_chunk_words: List[str] = []
            
            for para in paragraphs:
                para_words = para.split()
                
                # If paragraph itself is too large, split into sentences
                if len(para_words) > self.target_word_size:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    for sent in sentences:
                        sent_words = sent.split()
                        if not sent_words:
                            continue
                        if len(current_chunk_words) + len(sent_words) > self.target_word_size and current_chunk_words:
                            # Emit chunk
                            chunk_text = " ".join(current_chunk_words)
                            chunks.append(self._create_chunk_dict(
                                chunk_idx=chunk_idx,
                                paper_id=paper_id,
                                title=title,
                                authors=authors,
                                year=year,
                                section=sec_title,
                                page_start=page_start,
                                page_end=page_end,
                                text=chunk_text
                            ))
                            chunk_idx += 1
                            # Overlap
                            current_chunk_words = current_chunk_words[-self.overlap_word_size:] + sent_words
                        else:
                            current_chunk_words.extend(sent_words)
                else:
                    if len(current_chunk_words) + len(para_words) > self.target_word_size and current_chunk_words:
                        # Emit chunk
                        chunk_text = " ".join(current_chunk_words)
                        chunks.append(self._create_chunk_dict(
                            chunk_idx=chunk_idx,
                            paper_id=paper_id,
                            title=title,
                            authors=authors,
                            year=year,
                            section=sec_title,
                            page_start=page_start,
                            page_end=page_end,
                            text=chunk_text
                        ))
                        chunk_idx += 1
                        # Overlap
                        current_chunk_words = current_chunk_words[-self.overlap_word_size:] + para_words
                    else:
                        current_chunk_words.extend(para_words)

            # Flush remaining words in section
            if current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
                chunks.append(self._create_chunk_dict(
                    chunk_idx=chunk_idx,
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    section=sec_title,
                    page_start=page_start,
                    page_end=page_end,
                    text=chunk_text
                ))
                chunk_idx += 1

        return chunks

    def _create_chunk_dict(
        self,
        chunk_idx: int,
        paper_id: str,
        title: str,
        authors: List[str],
        year: Optional[int],
        section: str,
        page_start: int,
        page_end: int,
        text: str
    ) -> Dict[str, Any]:
        estimated_tokens = int(len(text.split()) * 1.3)
        return {
            "chunk_index": chunk_idx,
            "paper_id": paper_id,
            "paper_title": title,
            "authors": authors,
            "year": year,
            "section": section,
            "page_start": page_start,
            "page_end": page_end,
            "token_count": estimated_tokens,
            "text": text
        }
