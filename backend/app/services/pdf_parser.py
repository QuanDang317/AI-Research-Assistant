import re
import fitz  # PyMuPDF
from typing import Dict, List, Any, Optional
from pathlib import Path


class PDFParser:
    """
    Parser for scientific research papers (PDF) with multi-column layout support,
    section detection, and metadata extraction.
    """

    COMMON_SECTIONS = [
        "abstract",
        "introduction",
        "related work",
        "background",
        "method",
        "methodology",
        "proposed method",
        "system architecture",
        "experiments",
        "experimental setup",
        "results",
        "results and discussion",
        "discussion",
        "conclusion",
        "conclusions",
        "future work",
        "acknowledgments",
        "references"
    ]

    def __init__(self):
        pass

    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Main entrypoint to parse a PDF file.
        Returns extracted metadata, structured sections, and per-page text.
        """
        doc = fitz.open(file_path)
        total_pages = len(doc)

        metadata = self._extract_metadata(doc)
        pages_content = []
        all_blocks = []

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            page_blocks = self._extract_sorted_page_blocks(page, page_num)
            page_text = "\n\n".join([b["text"] for b in page_blocks if b["text"].strip()])

            pages_content.append({
                "page": page_num,
                "text": page_text
            })
            all_blocks.extend(page_blocks)

        # Segment blocks into sections
        sections = self._segment_into_sections(all_blocks, metadata.get("abstract", ""))

        return {
            "title": metadata["title"],
            "authors": metadata["authors"],
            "abstract": metadata["abstract"],
            "total_pages": total_pages,
            "sections": sections,
            "pages": pages_content
        }

    def _extract_sorted_page_blocks(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """
        Sorts text blocks respecting two-column academic paper layouts,
        and filters out running headers / footers.
        """
        rect = page.rect
        page_width = rect.width
        page_height = rect.height

        # Standard header/footer margin thresholds (points)
        header_limit = 45
        footer_limit = page_height - 45

        # Extract blocks: (x0, y0, x1, y1, text, block_no, block_type)
        raw_blocks = page.get_text("blocks")
        text_blocks = []

        for b in raw_blocks:
            # type 0 is text block
            if b[6] == 0:
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4].strip()
                if not text:
                    continue

                # Filter out obvious header/footer
                if y1 < header_limit or y0 > footer_limit:
                    # Ignore page numbers or header lines
                    if len(text) < 80 or text.isdigit():
                        continue

                # Determine if block is in left column, right column, or spans full width
                mid_x = page_width / 2.0
                if x1 <= mid_x + 30:
                    col = 0  # Left column
                elif x0 >= mid_x - 30:
                    col = 1  # Right column
                else:
                    col = -1 # Spans full width (e.g. Title, Abstract)

                text_blocks.append({
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "col": col,
                    "text": text,
                    "page": page_num
                })

        # Sorting: Full width blocks first (by y0), then Left column by y0, then Right column by y0
        def sort_key(block):
            if block["col"] == -1:
                return (0, block["y0"])
            elif block["col"] == 0:
                return (1, block["y0"])
            else:
                return (2, block["y0"])

        text_blocks.sort(key=sort_key)
        return text_blocks

    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """
        Extract title, authors, and abstract from document metadata and the first page.
        """
        meta = doc.metadata or {}
        title = meta.get("title", "").strip()
        author_str = meta.get("author", "").strip()

        authors = [a.strip() for a in re.split(r"[,;]", author_str) if a.strip()] if author_str else []
        abstract = ""

        if len(doc) > 0:
            first_page = doc[0]
            first_page_text = first_page.get_text("text")

            # Try to extract Title if not in metadata or if default placeholder
            if not title or len(title) < 5 or "untitled" in title.lower():
                # Attempt to get largest font size text on page 1
                title = self._detect_title_from_page1(first_page)

            # Try to extract Abstract
            abstract_match = re.search(r"(?i)abstract[:\s\n]+(.*?)(?=\n\s*(?:1[\.\s]|introduction|keywords|index terms))", first_page_text, re.DOTALL)
            if abstract_match:
                abstract = re.sub(r"\s+", " ", abstract_match.group(1).strip())

        return {
            "title": title or "Untitled Document",
            "authors": authors,
            "abstract": abstract
        }

    def _detect_title_from_page1(self, page: fitz.Page) -> str:
        """
        Finds text with the largest font size in the top half of page 1.
        """
        page_dict = page.get_text("dict")
        max_size = 0.0
        candidate_spans = []

        for block in page_dict.get("blocks", []):
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        size = span.get("size", 0.0)
                        bbox = span.get("bbox", [0, 0, 0, 0])

                        # Usually title is in top 40% of page 1
                        if bbox[1] < page.rect.height * 0.4 and len(text) > 2:
                            if size > max_size:
                                max_size = size
                                candidate_spans = [text]
                            elif abs(size - max_size) < 0.5:
                                candidate_spans.append(text)

        title = " ".join(candidate_spans)
        return re.sub(r"\s+", " ", title).strip() or "Untitled Document"

    def _segment_into_sections(self, blocks: List[Dict[str, Any]], abstract: str) -> List[Dict[str, Any]]:
        """
        Groups blocks into identifiable academic sections.
        """
        sections = []
        current_section_title = "Abstract" if abstract else "Preamble"
        current_section_blocks = []
        current_start_page = 1

        section_regex = re.compile(
            r"^(?:(?:\d+[\.\s]+|[IVXLCDM]+[\.\s]+))?(" + "|".join(self.COMMON_SECTIONS) + r")\b",
            re.IGNORECASE
        )

        for block in blocks:
            text = block["text"]
            first_line = text.split("\n")[0].strip()

            match = section_regex.match(first_line)
            if match and len(first_line) < 60:
                # Flush previous section
                if current_section_blocks:
                    content = "\n\n".join([b["text"] for b in current_section_blocks])
                    sections.append({
                        "section_title": current_section_title,
                        "page_start": current_start_page,
                        "page_end": current_section_blocks[-1]["page"],
                        "content": content
                    })
                    current_section_blocks = []

                current_section_title = first_line.title()
                current_start_page = block["page"]
                current_section_blocks.append(block)
            else:
                current_section_blocks.append(block)

        # Flush final section
        if current_section_blocks:
            content = "\n\n".join([b["text"] for b in current_section_blocks])
            sections.append({
                "section_title": current_section_title,
                "page_start": current_start_page,
                "page_end": current_section_blocks[-1]["page"],
                "content": content
            })

        return sections
