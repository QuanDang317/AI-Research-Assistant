import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.chat import ChatRequest, ChatResponse, Citation
from app.services.retriever import HybridRetriever
from app.services.generator import Generator

router = APIRouter(prefix="/api/v1", tags=["chat"])

retriever = HybridRetriever()
generator = Generator()


@router.post("/chat", response_model=ChatResponse)
def chat_with_papers(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Q&A endpoint: Retrieves relevant chunks using Hybrid Search & Reranking,
    generates answer with grounded citations, and records conversation history.
    """
    # 1. Manage Conversation
    conv = None
    if request.conversation_id:
        conv = db.query(Conversation).filter(Conversation.id == request.conversation_id).first()

    if not conv:
        conv = Conversation(
            id=request.conversation_id or str(uuid.uuid4()),
            title=request.query[:80] + ("..." if len(request.query) > 80 else "")
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)

    # 2. Retrieve Conversation History
    prev_messages = (
        db.query(Message)
        .filter(Message.conversation_id == conv.id)
        .order_by(Message.created_at.asc())
        .limit(6)
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in prev_messages]

    # 3. Hybrid Retrieval & Rerank
    filters_dict = {}
    if request.filters:
        if request.filters.year_min is not None:
            filters_dict["year_min"] = request.filters.year_min
        if request.filters.year_max is not None:
            filters_dict["year_max"] = request.filters.year_max
        if request.filters.fields:
            filters_dict["field"] = request.filters.fields[0]

    if request.paper_ids:
        filters_dict["paper_ids"] = request.paper_ids

    retrieved_chunks = retriever.retrieve(
        query=request.query,
        filters=filters_dict if filters_dict else None,
        limit=request.top_k,
        apply_rerank=True
    )

    # 4. Generate Grounded Answer
    gen_result = generator.generate_answer(
        query=request.query,
        retrieved_chunks=retrieved_chunks,
        conversation_history=history
    )

    # 5. Save Messages to Database
    user_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="user",
        content=request.query
    )
    db.add(user_msg)

    chunk_ids = [c["id"] for c in retrieved_chunks]
    asst_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="assistant",
        content=gen_result["answer"],
        citations=gen_result["citations"],
        chunks_used=chunk_ids
    )
    db.add(asst_msg)
    db.commit()

    citations_resp = [
        Citation(
            paper_id=c.get("paper_id") or "",
            paper_title=c.get("paper_title") or "Unknown Paper",
            section=c.get("section"),
            page=c.get("page"),
            chunk_text=c.get("chunk_text") or "",
            relevance_score=c.get("relevance_score")
        )
        for c in gen_result["citations"]
    ]

    return ChatResponse(
        conversation_id=conv.id,
        answer=gen_result["answer"],
        citations=citations_resp,
        metadata={
            "chunks_retrieved": len(retrieved_chunks),
            "faithfulness": gen_result.get("faithfulness", "High")
        }
    )


@router.get("/conversations")
def list_conversations(db: Session = Depends(get_db)):
    """
    List all conversations.
    """
    return db.query(Conversation).order_by(Conversation.updated_at.desc()).all()


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """
    Get conversation details and full message history.
    """
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    return {
        "conversation": conv,
        "messages": messages
    }


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """
    Delete a conversation and its messages.
    """
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conv)
    db.commit()
    return {"message": "Conversation deleted successfully"}
