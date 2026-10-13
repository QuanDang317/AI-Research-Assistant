import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fitz
from app.db.session import init_db, SessionLocal
from app.api.papers import parser, chunker, embedder
from app.models.paper import Paper
from app.models.chunk import Chunk
from app.db.qdrant import get_qdrant_client, upsert_chunks_to_qdrant
import uuid
import hashlib

def create_pdf(filepath: str, title: str, authors: str, abstract: str, col1: str, col2: str):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    page.insert_text((72, 60), title, fontsize=16)
    page.insert_text((72, 85), f"Authors: {authors}", fontsize=10)
    page.insert_text((72, 115), f"Abstract — {abstract}", fontsize=9)

    page.insert_textbox(fitz.Rect(72, 150, 280, 720), col1, fontsize=8.5)
    page.insert_textbox(fitz.Rect(320, 150, 540, 720), col2, fontsize=8.5)

    doc.save(filepath)
    doc.close()

def seed_sample_papers():
    print("[*] Seeding sample research papers...")
    init_db()
    db = SessionLocal()
    qdrant = get_qdrant_client()

    os.makedirs("./data/papers", exist_ok=True)

    papers_def = [
        {
            "filename": "paper_resnet_crop.pdf",
            "title": "Deep Residual Learning for Agricultural Disease Recognition",
            "authors": "Nguyen Van A, Tran Thi B, Le Van C",
            "year": 2023,
            "field": "Computer Vision",
            "abstract": "This study develops a high-performance deep residual network (ResNet-50) for identifying 38 foliar disease categories across 54,306 images.",
            "col1": (
                "1. Introduction\n"
                "Early plant disease recognition is paramount to global food security. "
                "Traditional methods fail in challenging natural field lighting conditions.\n\n"
                "2. Methodology\n"
                "We adopt a pretrained ResNet-50 backbone with skip connections. "
                "Data augmentation includes random color jittering, rotation, and CutMix. "
                "Training is performed using SGD with momentum 0.9 and learning rate 0.001."
            ),
            "col2": (
                "3. Experimental Results\n"
                "On the PlantVillage benchmark dataset, our ResNet-50 achieves 95.8% top-1 accuracy "
                "and an F1-score of 95.2%. Training converges within 40 epochs.\n\n"
                "4. Conclusion\n"
                "Residual connections effectively mitigate vanishing gradients, allowing deeper networks "
                "to generalize well to agricultural pathology."
            )
        },
        {
            "filename": "paper_vit_crop.pdf",
            "title": "Vision Transformers for Crop Pathology Diagnostics",
            "authors": "Smith J, Williams D, Davis K",
            "year": 2024,
            "field": "Machine Learning",
            "abstract": "We explore Vision Transformers (ViT-B/16) for self-attention based feature extraction in agricultural image classification.",
            "col1": (
                "1. Introduction\n"
                "While CNNs emphasize local inductive biases, Vision Transformers capture long-range contextual relationships.\n\n"
                "2. Methodology\n"
                "Images of resolution 224x224 are divided into 16x16 non-overlapping patches. "
                "Each patch is projected to 768-dimensional token vectors and prepended with a class token. "
                "We train with AdamW optimizer and cosine learning rate decay."
            ),
            "col2": (
                "3. Experimental Results\n"
                "ViT-B/16 demonstrates 97.1% accuracy on PlantVillage, outperforming baseline ResNet by 1.3%. "
                "Attention heatmaps confirm focus on disease lesions.\n\n"
                "4. Conclusion\n"
                "Transformers provide superior global representation learning for complex crop leaf spots."
            )
        },
        {
            "filename": "paper_efficientnet_crop.pdf",
            "title": "EfficientNet Scaling for Edge Agricultural Devices",
            "authors": "Tan M, Le Q",
            "year": 2024,
            "field": "Edge AI",
            "abstract": "This paper presents EfficientNet-B4 compound scaling for low-latency edge deployment on agricultural drones.",
            "col1": (
                "1. Introduction\n"
                "Deploying heavy models on edge farm devices requires balancing parameter count, FLOPS, and accuracy.\n\n"
                "2. Methodology\n"
                "We apply compound scaling to balance depth, width, and input image resolution. "
                "MBConv blocks with Squeeze-and-Excitation optimization are utilized."
            ),
            "col2": (
                "3. Experimental Results\n"
                "EfficientNet-B4 achieves 96.8% accuracy with 4.2x fewer parameters than ResNet-101. "
                "Inference latency on mobile chipsets is 24ms per frame.\n\n"
                "4. Conclusion\n"
                "Compound scaling yields the optimal trade-off for real-time robotic weed and disease spraying."
            )
        }
    ]

    for p in papers_def:
        filepath = os.path.join("./data/papers", p["filename"])
        create_pdf(filepath, p["title"], p["authors"], p["abstract"], p["col1"], p["col2"])

        # Check existing in DB
        with open(filepath, "rb") as f:
            f_bytes = f.read()
        f_hash = hashlib.sha256(f_bytes).hexdigest()

        existing = db.query(Paper).filter(Paper.file_hash == f_hash).first()
        if existing:
            print(f"[-] Paper already seeded: {p['title']}")
            continue

        paper_id = str(uuid.uuid4())
        paper = Paper(
            id=paper_id,
            title=p["title"],
            authors=[a.strip() for a in p["authors"].split(",")],
            year=p["year"],
            field=p["field"],
            abstract=p["abstract"],
            file_path=filepath,
            file_hash=f_hash,
            total_pages=1,
            status="ready"
        )
        db.add(paper)
        db.commit()

        # Parse & Chunk & Vectorize
        parsed = parser.parse(filepath)
        chunks_data = chunker.chunk_paper(parsed, paper_id=paper.id)

        if chunks_data:
            embs = embedder.embed_texts([c["text"] for c in chunks_data])
            for c in chunks_data:
                c["id"] = str(uuid.uuid4())
                c["qdrant_id"] = c["id"]
            upsert_chunks_to_qdrant(chunks_data, embs, client=qdrant)

            for c in chunks_data:
                chunk_rec = Chunk(
                    id=c["id"],
                    paper_id=paper.id,
                    chunk_index=c["chunk_index"],
                    section=c["section"],
                    page_start=c["page_start"],
                    page_end=c["page_end"],
                    text=c["text"],
                    token_count=c["token_count"],
                    qdrant_id=c["qdrant_id"]
                )
                db.add(chunk_rec)
            db.commit()

        print(f"[+] Successfully seeded: {p['title']} ({len(chunks_data)} chunks)")

    db.close()
    print("[+] All sample papers have been seeded successfully!")

if __name__ == "__main__":
    seed_sample_papers()
