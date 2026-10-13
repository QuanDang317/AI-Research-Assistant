import os
import uuid
import hashlib
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.qdrant import get_qdrant_client, upsert_chunks_to_qdrant
from app.models.paper import Paper
from app.models.chunk import Chunk
from app.schemas.paper import PaperResponse, ChunkResponse
from app.services.pdf_parser import PDFParser
from app.services.chunker import SemanticChunker
from app.services.embedder import Embedder
from app.config import settings
from qdrant_client.models import Filter, FieldCondition, MatchValue

router = APIRouter(prefix="/api/v1/papers", tags=["papers"])

parser = PDFParser()
chunker = SemanticChunker(target_word_size=400, overlap_word_size=50)
embedder = Embedder()


def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


@router.post("/upload", response_model=PaperResponse)
async def upload_paper(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a research paper PDF, extract text & metadata,
    perform semantic chunking, embed vectors, and store in
    both Relational DB and Vector DB (Qdrant).
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    file_hash = compute_file_hash(content)

    # Check duplicate
    existing_paper = db.query(Paper).filter(Paper.file_hash == file_hash).first()
    if existing_paper:
        return existing_paper

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    paper_id = str(uuid.uuid4())
    saved_filename = f"{paper_id}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, saved_filename)

    with open(file_path, "wb") as f:
        f.write(content)

    # 1. Parse PDF
    try:
        parsed_doc = parser.parse(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")

    # 2. Create Paper record
    paper = Paper(
        id=paper_id,
        title=parsed_doc.get("title") or file.filename,
        authors=parsed_doc.get("authors") or [],
        abstract=parsed_doc.get("abstract") or "",
        total_pages=parsed_doc.get("total_pages", 1),
        file_path=file_path,
        file_hash=file_hash,
        status="ready"
    )
    db.add(paper)
    db.commit()
    db.refresh(paper)

    # 3. Semantic Chunking
    chunks_data = chunker.chunk_paper(parsed_doc, paper_id=paper.id)

    if chunks_data:
        # 4. Generate Embeddings (Dense + Sparse)
        texts_to_embed = [c["text"] for c in chunks_data]
        embeddings = embedder.embed_texts(texts_to_embed)

        # Assign unique UUID for each chunk (shared between DB and Qdrant)
        for c in chunks_data:
            c["id"] = str(uuid.uuid4())
            c["qdrant_id"] = c["id"]

        # 5. Store in Qdrant Vector DB
        try:
            upsert_chunks_to_qdrant(chunks=chunks_data, embeddings=embeddings)
        except Exception as e:
            # Non-blocking warning if vector DB has issues
            pass

        # 6. Store Chunks in PostgreSQL / SQLite
        chunk_records = []
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
            chunk_records.append(chunk_rec)

        db.add_all(chunk_records)
        db.commit()

    return paper


@router.get("", response_model=List[PaperResponse])
def list_papers(db: Session = Depends(get_db)):
    """
    List all uploaded papers.
    """
    return db.query(Paper).order_by(Paper.created_at.desc()).all()


@router.get("/{paper_id}", response_model=PaperResponse)
def get_paper(paper_id: str, db: Session = Depends(get_db)):
    """
    Get paper metadata by ID.
    """
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper


@router.get("/{paper_id}/chunks", response_model=List[ChunkResponse])
def get_paper_chunks(paper_id: str, db: Session = Depends(get_db)):
    """
    Get all semantic chunks for a given paper.
    """
    chunks = db.query(Chunk).filter(Chunk.paper_id == paper_id).order_by(Chunk.chunk_index.asc()).all()
    return chunks


@router.delete("/{paper_id}")
def delete_paper(paper_id: str, db: Session = Depends(get_db)):
    """
    Delete a paper and all associated chunks, physical file,
    and Qdrant vector points.
    """
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    # 1. Remove physical file
    if os.path.exists(paper.file_path):
        try:
            os.remove(paper.file_path)
        except OSError:
            pass

    # 2. Remove from Qdrant Vector DB
    try:
        qdrant = get_qdrant_client()
        qdrant.delete(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="paper_id",
                        match=MatchValue(value=paper_id)
                    )
                ]
            )
        )
    except Exception:
        pass

    # 3. Delete from Relational DB
    db.delete(paper)
    db.commit()
    return {"message": f"Paper '{paper.title}' successfully deleted."}
