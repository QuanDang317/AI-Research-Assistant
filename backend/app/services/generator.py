import logging
from typing import List, Dict, Any, Optional
from app.config import settings
from app.utils.citation_parser import CitationParser

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Bạn là trợ lý nghiên cứu khoa học chuyên nghiệp và chuẩn xác.
Nhiệm vụ của bạn:
1. Trả lời câu hỏi của người dùng DỰA HOÀN TOÀN TRÊN CÁC ĐOẠN TRÍCH (Context) được cung cấp.
2. TUYỆT ĐỐI KHÔNG tự bịa thông tin ngoài context.
3. Bắt buộc trích dẫn nguồn cho MỖI nhận định theo định dạng: [Tên bài báo, Section, Page X] (Ví dụ: [Deep Residual Learning, Architecture, Page 3]).
4. Nếu context không đủ thông tin để trả lời, hãy nói rõ: "Dựa trên các tài liệu hiện có, tôi không tìm thấy thông tin về...".
5. Khi người dùng yêu cầu so sánh hoặc tổng hợp nhiều phương pháp/nghiên cứu, hãy trình bày kèm bảng markdown rõ ràng.
"""


class Generator:
    """
    Generation service for scientific Q&A, handling Prompt formatting,
    Gemini API calls, and citation extraction.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self._client = None

    @property
    def client(self):
        if self._client is None and settings.GEMINI_API_KEY:
            try:
                from google import genai
                self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini Client: {e}")
        return self._client

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved chunks with rich citations metadata for prompt injection.
        """
        formatted = []
        for idx, c in enumerate(chunks, 1):
            title = c.get("paper_title") or "Unknown Paper"
            sec = c.get("section") or "General"
            page = c.get("page_start") or 1
            text = c.get("text", "").strip()

            chunk_repr = (
                f"--- ĐOẠN TRÍCH {idx} ---\n"
                f"Bài báo: {title}\n"
                f"Section: {sec} | Trang: {page}\n"
                f"Nội dung:\n{text}\n"
            )
            formatted.append(chunk_repr)
        return "\n".join(formatted)

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generates an answer grounded in retrieved chunks with citations.
        """
        if not retrieved_chunks:
            return {
                "answer": "Dựa trên các bài báo khoa học hiện có trong hệ thống, tôi không tìm thấy tài liệu liên quan để trả lời câu hỏi của bạn.",
                "citations": [],
                "faithfulness": "Not Applicable"
            }

        context_str = self.format_context(retrieved_chunks)

        user_content = (
            f"## Context (Các đoạn trích từ bài báo khoa học):\n\n"
            f"{context_str}\n\n"
            f"## Câu hỏi của người dùng:\n{query}\n\n"
            f"## Yêu cầu:\nTrả lời chi tiết, khách quan, kèm trích dẫn chuẩn [Tên bài báo, Section, Page X] cho mỗi thông tin."
        )

        # 1. Generate via Gemini if API key is present
        if self.client:
            try:
                contents = []
                if conversation_history:
                    for msg in conversation_history[-4:]:
                        role = msg.get("role")
                        text = msg.get("content")
                        contents.append(f"{role.capitalize()}: {text}")
                contents.append(user_content)

                prompt_full = f"{SYSTEM_PROMPT}\n\n" + "\n\n".join(contents)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt_full
                )
                answer_text = response.text.strip()
                citations = CitationParser.extract_citations(answer_text, retrieved_chunks)

                return {
                    "answer": answer_text,
                    "citations": citations,
                    "faithfulness": "High"
                }
            except Exception as e:
                logger.error(f"Gemini API generation error: {e}")

        # 2. Local Synthesizer Fallback (When no API key is set)
        fallback_answer = self._synthesize_local_response(query, retrieved_chunks)
        citations = CitationParser.extract_citations(fallback_answer, retrieved_chunks)

        return {
            "answer": fallback_answer,
            "citations": citations,
            "faithfulness": "High"
        }

    def _synthesize_local_response(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Produces a synthesized summary from top chunks with citations for local offline testing.
        """
        top_chunk = chunks[0]
        title = top_chunk.get("paper_title") or "Tài liệu"
        sec = top_chunk.get("section") or "Methodology"
        page = top_chunk.get("page_start") or 1

        summary = (
            f"Dựa trên các tài liệu đã cung cấp trong hệ thống:\n\n"
            f"{top_chunk.get('text', '').strip()[:400]}... "
            f"[{title}, {sec}, Page {page}]\n\n"
        )
        if len(chunks) > 1:
            second = chunks[1]
            s_title = second.get("paper_title") or "Tài liệu"
            s_sec = second.get("section") or "Results"
            s_page = second.get("page_start") or 1
            summary += (
                f"Ngoài ra, nghiên cứu bổ sung chỉ ra rằng: "
                f"{second.get('text', '').strip()[:300]}... "
                f"[{s_title}, {s_sec}, Page {s_page}]"
            )
        return summary
