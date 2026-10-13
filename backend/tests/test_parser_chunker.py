import os
import fitz
import pytest
from app.services.pdf_parser import PDFParser
from app.services.chunker import SemanticChunker

def create_sample_academic_pdf(filepath: str):
    """
    Creates a sample 2-column academic PDF with Title, Abstract, and Sections.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # Standard Letter size

    # Title (large font, centered top)
    page.insert_text((72, 70), "Deep Learning for Plant Disease Detection", fontsize=18)
    page.insert_text((72, 95), "Authors: Nguyen Van A, Tran Thi B", fontsize=11)

    # Abstract (full width)
    page.insert_text((72, 130), "Abstract - This paper presents an automated approach to crop disease identification using deep neural networks. We achieve 97.5% accuracy.", fontsize=10)

    # Column 1 (Left: x in [72, 280])
    col1_text = (
        "1. Introduction\n"
        "Plant diseases pose a severe threat to global food security. "
        "Early and accurate diagnosis is critical to preventing large-scale crop loss. "
        "Traditional visual inspection by agricultural experts is labor-intensive and subjective.\n\n"
        "2. Methodology\n"
        "We employ a ResNet-50 backbone fine-tuned on the PlantVillage dataset. "
        "Images were augmented using random rotations, flips, and color jittering. "
        "We train with Adam optimizer and cosine annealing schedule."
    )
    page.insert_textbox(fitz.Rect(72, 170, 280, 700), col1_text, fontsize=9)

    # Column 2 (Right: x in [320, 540])
    col2_text = (
        "3. Experimental Results\n"
        "Our model was evaluated on 54,306 images across 38 disease categories. "
        "We observed a top-1 accuracy of 97.5% and an F1-score of 96.8%. "
        "Compared to baseline VGG-16, our approach improves inference speed by 35%.\n\n"
        "4. Conclusion\n"
        "In this work, we demonstrated the efficacy of residual learning for agricultural diagnostics. "
        "Future directions include edge deployment on mobile devices."
    )
    page.insert_textbox(fitz.Rect(320, 170, 540, 700), col2_text, fontsize=9)

    doc.save(filepath)
    doc.close()

def test_pdf_parser_and_chunker(tmp_path):
    pdf_path = str(tmp_path / "sample_paper.pdf")
    create_sample_academic_pdf(pdf_path)

    # 1. Test Parser
    parser = PDFParser()
    parsed = parser.parse(pdf_path)

    assert "Plant Disease" in parsed["title"]
    assert "Abstract" in parsed["abstract"] or len(parsed["abstract"]) > 0
    assert parsed["total_pages"] == 1
    assert len(parsed["sections"]) >= 2

    # 2. Test Chunker
    chunker = SemanticChunker(target_word_size=50, overlap_word_size=10)
    chunks = chunker.chunk_paper(parsed, paper_id="test-paper-001")

    assert len(chunks) > 0
    assert chunks[0]["paper_id"] == "test-paper-001"
    assert "section" in chunks[0]
    assert chunks[0]["token_count"] > 0
