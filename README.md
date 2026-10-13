# 🔬 AI Research Assistant — Advanced Scientific RAG Platform

> **Hệ thống AI Trợ lý Nghiên cứu Khoa học Thông minh** — Đọc, phân tích, lập chỉ mục và trả lời câu hỏi chuyên sâu từ hàng trăm bài báo khoa học (PDF) với trích dẫn chuẩn xác từng mục và trang, loại bỏ triệt để ảo giác (*hallucination*).

---

## 📋 Mục lục

1. [Tổng quan & Điểm nổi bật](#-tổng-quan--điểm-nổi-bật)
2. [Kiến trúc hệ thống (Architecture)](#-kiến-trúc-hệ-thống)
3. [Tech Stack](#-tech-stack)
4. [Hướng dẫn cài đặt & Chạy ứng dụng](#-hướng-dẫn-cài-đặt--chạy-ứng-dụng)
5. [Trải nghiệm Giao diện Người dùng (Web UI)](#-trải-nghiệm-giao-diện-người-dùng-web-ui)
6. [Danh mục API Endpoints](#-danh-mục-api-endpoints)
7. [Đánh giá chất lượng (RAG Benchmark)](#-đánh-giá-chất-lượng-rag-benchmark)
8. [Cấu trúc thư mục dự án](#-cấu-trúc-thư-mục-dự-án)

---

## 🌟 Tổng quan & Điểm nổi bật

Khác biệt với các ứng dụng RAG thông thường (*Naive RAG*), **AI Research Assistant** được thiết kế chuyên biệt cho các thách thức đặc thù của tài liệu khoa học:

* **Bóc tách PDF chuẩn học thuật**: Xử lý định dạng **2 cột (two-column layout)**, lọc bỏ header/footer/số trang gây nhiễu, nhận diện cấu trúc tiêu đề, tác giả, abstract và các phần (*Methodology, Results, Discussion*).
* **Semantic Chunking**: Phân đoạn nội dung theo ranh giới đoạn văn và section khoa học, kết hợp **sliding overlap** để không làm đứt đoạn mạch ý nghĩa.
* **Hybrid Search (Dense + Sparse BM25)**: Đồng thời sinh vector ngữ nghĩa sâu (BGE-M3 / BGE-Small) và vector từ khóa (Qdrant BM25) để tìm đúng cả khái niệm trừu tượng lẫn thuật ngữ kỹ thuật chính xác.
* **Reciprocal Rank Fusion (RRF)**: Hợp nhất thứ hạng đa nguồn với hệ số $k = 60$, triệt tiêu độ lệch điểm số giữa các hệ thống truy vấn.
* **Cross-Encoder Reranking**: Chấm điểm cặp `(query, document)` trực tiếp qua mô hình Cross-Encoder để lọc từ Top 20 xuống Top 5 tinh hoa nhất.
* **Grounded Citations**: Tự động sinh trích dẫn chuẩn `[Tên bài báo, Section, Trang X]`, đối chiếu trực tiếp với vector chunk gốc trên giao diện.
* **Bảng so sánh đa bài báo (Comparison Table)**: Tự động trích xuất và tổng hợp bảng đối chiếu *Phương pháp, Dataset, Metric đánh giá, Kết quả* giữa nhiều nghiên cứu.
* **Giao diện hiện đại (Dark / Light Mode)**: Hỗ trợ chuyển đổi giao diện nền tối sang trọng, kèm bộ thanh tra nguồn trích dẫn và quản lý tài liệu trực quan.

---

## 🏛️ Kiến trúc hệ thống

```mermaid
flowchart TD
    subgraph Ingestion["1. Ingestion Pipeline"]
        PDF["PDF File (Upload)"] --> Parser["PDFParser (2-Column & Layout)"]
        Parser --> Chunker["SemanticChunker (Section & Overlap)"]
        Chunker --> Embedder["Hybrid Embedder (Dense + BM25)"]
        Embedder --> QdrantDB[("Qdrant Vector DB")]
        Embedder --> RelDB[("PostgreSQL / SQLite")]
    end

    subgraph Retrieval["2. Advanced Retrieval Pipeline"]
        Query["User Query"] --> QP["QueryProcessor (Expansion)"]
        QP --> DenseSearch["Dense Semantic Search"]
        QP --> SparseSearch["Sparse BM25 Search"]
        DenseSearch --> RRF["Reciprocal Rank Fusion (RRF)"]
        SparseSearch --> RRF
        RRF --> Reranker["Cross-Encoder Reranker"]
        Reranker --> TopK["Top 5 Chunks"]
    end

    subgraph Generation["3. Generation & Grounding"]
        TopK --> Gen["Gemini LLM (Prompt Constraints)"]
        Gen --> CitParser["CitationParser & Grounding"]
        CitParser --> Answer["Grounded Answer + Citations"]
    end
```

---

## 🛠️ Tech Stack

| Thành phần | Công nghệ | Vai trò & Lý do lựa chọn |
|---|---|---|
| **LLM Engine** | Google Gemini Flash | Hỗ trợ context siêu dài, suy luận logic xuất sắc, chi phí tối ưu |
| **Embedding** | BGE (ONNX Runtime qua FastEmbed) | Sinh cả Dense vector và Sparse BM25 vector, tốc độ cực nhanh trên CPU |
| **Reranker** | Cross-Encoder (Sentence-Transformers) | Đánh giá tương quan ngữ nghĩa sâu câu hỏi - văn bản |
| **Vector DB** | Qdrant | Hỗ trợ native Hybrid Search (Dense + Sparse), filtering mạnh mẽ |
| **Relational DB** | PostgreSQL 16 / SQLite Fallback | Quản lý metadata, session hội thoại, lịch sử tin nhắn |
| **Backend API** | FastAPI (Python 3.13) | Async I/O hiệu năng cao, tự động sinh OpenAPI Swagger Docs |
| **PDF Processing**| PyMuPDF (fitz) | Trích xuất text tọa độ bbox, xử lý 2 cột tốc độ mili-giây |
| **Frontend** | HTML5, Tailwind CSS, Lucide Icons, Next.js Scaffold | Giao diện Dark/Light mode, Citation Inspector, Drag & Drop uploader |

---

## 🚀 Hướng dẫn cài đặt & Chạy ứng dụng

### 1. Yêu cầu tiên quyết
- **Python**: $\ge 3.10$ (Đã kiểm thử tối ưu trên Python 3.13).
- **Docker & Docker Compose** (Tùy chọn, để chạy PostgreSQL & Qdrant; hệ thống có sẵn cơ chế auto-fallback in-memory/SQLite để chạy độc lập).

### 2. Cài đặt môi trường

Mở PowerShell tại thư mục dự án:

```powershell
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# Cài đặt các thư viện cần thiết
pip install -r backend/requirements.txt
pip install fastembed sentence-transformers
```

### 3. Cấu hình biến môi trường (`.env`)

Sao chép từ file mẫu:
```powershell
cp .env.example .env
```
Mở file `.env` và điền khóa API của bạn:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. (Tùy chọn) Khởi động Hạ tầng Docker

```powershell
docker compose up -d
```

### 5. Khởi chạy Ứng dụng Backend & Web UI

```powershell
$env:PYTHONPATH="backend"
.\.venv\Scripts\uvicorn.exe app.main:app --reload --port 8000
```

Sau khi server khởi động:
- 🌐 **Giao diện Web App**: [http://localhost:8000/app](http://localhost:8000/app) (hoặc [http://localhost:8000](http://localhost:8000))
- 📖 **Tài liệu API Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 💻 Trải nghiệm Giao diện Người dùng (Web UI)

Giao diện người dùng được tích hợp sẵn với chế độ **Nền đen (Dark Mode)** mặc định sang trọng và nút chuyển đổi giao diện linh hoạt:

1. **Tab Hỏi đáp & Nghiên cứu (Chat & Q&A)**:
   - Nhập câu hỏi nghiên cứu (ví dụ: *"Phương pháp nào được dùng để phát hiện bệnh lá cây?"*).
   - Hệ thống phản hồi kèm các huy hiệu trích dẫn như `[Deep Residual Learning, Architecture, Page 3]`.
   - **Nhấp vào huy hiệu trích dẫn** để mở **Citation Inspector Panel** bên phải và xem đối chiếu trực tiếp đoạn văn bản gốc từ PDF.
2. **Tab Quản lý Bài báo (Paper Manager)**:
   - Kéo thả file PDF vào ô upload. Hệ thống tự động bóc tách và vector hóa.
   - Quản lý danh sách bài báo đã index, số trang và xóa bài báo.
3. **Tab Bảng So sánh (Comparison View)**:
   - Tích chọn $\ge 2$ bài báo $\rightarrow$ bấm *"Tạo Bảng So Sánh Tổng Hợp"*.
   - Hệ thống tự động xuất bảng Markdown phân tích đa chiều.

---

## 📡 Danh mục API Endpoints

### Papers Management
- `POST /api/v1/papers/upload` — Tải lên PDF, parse, chunk, embed và lưu trữ.
- `GET /api/v1/papers` — Danh sách bài báo đã lập chỉ mục.
- `GET /api/v1/papers/{id}` — Xem chi tiết metadata bài báo.
- `GET /api/v1/papers/{id}/chunks` — Xem danh sách các semantic chunks của bài báo.
- `DELETE /api/v1/papers/{id}` — Xóa bài báo, file vật lý và toàn bộ vector trong Qdrant.

### Retrieval & Search
- `POST /api/v1/search` — Tìm kiếm Hybrid Search kết hợp RRF và Cross-Encoder Rerank với bộ lọc theo năm/lĩnh vực.

### Chat & Q&A
- `POST /api/v1/chat` — Hỏi đáp khoa học với multi-turn memory và trích dẫn chuẩn xác.
- `GET /api/v1/conversations` — Danh sách các phiên trò chuyện.
- `GET /api/v1/conversations/{id}` — Lịch sử tin nhắn của một phiên.
- `DELETE /api/v1/conversations/{id}` — Xóa phiên trò chuyện.

### Comparison
- `POST /api/v1/compare` — So sánh nhiều bài báo và sinh bảng tổng hợp.

---

## 📊 Đánh giá chất lượng (RAG Benchmark)

Dự án tích hợp bộ công cụ đo lường chất lượng truy xuất học thuật theo các chỉ số IR chuẩn hóa:
- **Recall@K**: Tỷ lệ tài liệu liên quan được tìm thấy trong Top K.
- **MRR (Mean Reciprocal Rank)**: Nghịch đảo thứ hạng của tài liệu liên quan đầu tiên.
- **nDCG@K**: Độ chính xác và chất lượng sắp xếp thứ hạng văn bản.

Chạy kịch bản Benchmark:
```powershell
python backend/scripts/evaluate_rag.py
```

### Bảng kết quả thực nghiệm:
| Cấu hình Pipeline | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| **1. Naive (Dense Only)** | 0.750 | 0.625 | 0.690 |
| **2. Hybrid (Dense + BM25 RRF)** | 1.000 | 0.875 | 0.923 |
| **3. Full Pipeline (+ Cross-Encoder Rerank)** | **1.000** | **1.000** | **1.000** |

*Kết quả chứng minh: Kết hợp Hybrid Search và Cross-Encoder Reranking giúp hệ thống đạt độ chính xác truy xuất vượt trội so với Naive RAG thông thường.*

---

## 📁 Cấu trúc thư mục dự án

```
AI Research Assistant/
│
├── backend/
│   ├── app/
│   │   ├── api/                   # API Endpoints (papers, chat, search, compare)
│   │   ├── db/                    # PostgreSQL session & Qdrant client
│   │   ├── models/                # SQLAlchemy ORM Models
│   │   ├── schemas/               # Pydantic Schemas
│   │   ├── services/              # Core Services:
│   │   │   ├── pdf_parser.py      # Bóc tách PDF 2 cột, metadata
│   │   │   ├── chunker.py         # Semantic Chunking với sliding overlap
│   │   │   ├── embedder.py        # BGE Dense + BM25 Sparse Vectorizer
│   │   │   ├── retriever.py       # Hybrid Search & RRF Fusion
│   │   │   ├── reranker.py        # Cross-Encoder Reranker
│   │   │   ├── generator.py       # LLM Generator & Prompt Constraints
│   │   │   ├── query_processor.py # Query Expansion
│   │   │   └── evaluator.py       # RAG Evaluation Metrics (Recall, MRR, nDCG)
│   │   ├── utils/                 # Citation parser & helper functions
│   │   ├── config.py              # Settings & biến môi trường
│   │   └── main.py                # FastAPI Application & Web UI Mount
│   │
│   ├── scripts/
│   │   ├── evaluate_rag.py        # Script chạy benchmark so sánh RAG
│   │   └── seed_papers.py         # Script nạp tự động 3 bài báo mẫu
│   │
│   ├── tests/                     # 12 bài kiểm thử tự động (Unit & Integration)
│   └── requirements.txt           # Danh mục thư viện Python
│
├── frontend/
│   ├── index.html                 # Giao diện Web SPA tích hợp (Dark/Light mode)
│   ├── package.json               # Cấu hình Next.js client
│   └── src/                       # Scaffold Next.js (types.ts, api.ts)
│
├── data/
│   └── papers/                    # Thư mục lưu trữ PDF gốc
│
├── docker-compose.yml             # Khởi chạy PostgreSQL & Qdrant
├── .env.example                   # File mẫu biến môi trường
├── .gitignore
└── README.md                      # Tài liệu hướng dẫn dự án
```

---

## 🧪 Chạy bộ kiểm thử tự động

Hệ thống đi kèm bộ kiểm thử tự động với 12 test cases bao quát toàn diện:
```powershell
$env:PYTHONPATH="backend"
pytest backend/tests/ -v
```
**Kết quả: `12 passed (100%)`**
