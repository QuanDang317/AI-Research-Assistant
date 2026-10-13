# 🔬 AI Research Assistant — Blueprint & Technical Guide

> **Trợ lý nghiên cứu khoa học thông minh** — Hệ thống AI đọc, phân tích và trả lời câu hỏi
> từ hàng trăm bài báo khoa học, kèm trích dẫn nguồn chính xác.

---

## 📋 Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Vấn đề cần giải quyết](#2-vấn-đề-cần-giải-quyết)
3. [Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
4. [Tech Stack chi tiết](#4-tech-stack-chi-tiết)
5. [RAG Pipeline — Giải thích từng bước](#5-rag-pipeline--giải-thích-từng-bước)
   - 5.1 [Ingestion Pipeline (Nạp dữ liệu)](#51-ingestion-pipeline-nạp-dữ-liệu)
   - 5.2 [Retrieval Pipeline (Truy xuất)](#52-retrieval-pipeline-truy-xuất)
   - 5.3 [Generation Pipeline (Sinh câu trả lời)](#53-generation-pipeline-sinh-câu-trả-lời)
6. [Các kỹ thuật nâng cao](#6-các-kỹ-thuật-nâng-cao)
7. [Database Schema](#7-database-schema)
8. [API Endpoints](#8-api-endpoints)
9. [Cấu trúc thư mục](#9-cấu-trúc-thư-mục)
10. [Kế hoạch triển khai theo Phase](#10-kế-hoạch-triển-khai-theo-phase)
11. [Đánh giá chất lượng (RAG Evaluation)](#11-đánh-giá-chất-lượng-rag-evaluation)
12. [Deployment](#12-deployment)
13. [Rủi ro & Giải pháp](#13-rủi-ro--giải-pháp)
14. [Tài liệu tham khảo](#14-tài-liệu-tham-khảo)

---

## 1. Tổng quan dự án

### Mô tả

AI Research Assistant là một hệ thống **Retrieval-Augmented Generation (RAG)** chuyên biệt
cho tài liệu khoa học. Người dùng upload các bài báo (PDF), hệ thống sẽ:

- **Đọc & phân tích** nội dung paper (text, bảng, hình ảnh)
- **Lưu trữ** dưới dạng vector embeddings + metadata
- **Trả lời câu hỏi** dựa trên nội dung papers với trích dẫn chính xác
- **So sánh** nhiều papers, tạo bảng tổng hợp kết quả

### Ví dụ sử dụng

```
👤 User: "Những phương pháp nào được sử dụng để phát hiện bệnh trên lá cây?"

🤖 Assistant:
Dựa trên các nghiên cứu đã cung cấp, có 3 nhóm phương pháp chính:

1. **CNN-based methods**: Sử dụng các kiến trúc như ResNet-50 và VGG-16
   để phân loại ảnh lá cây bị bệnh. [Paper A, Section 3.1, p.5]

2. **Vision Transformer (ViT)**: Áp dụng self-attention mechanism cho
   image patches, đạt accuracy 96.2% trên PlantVillage dataset.
   [Paper B, Table 2, p.8]

3. **Hybrid approaches**: Kết hợp CNN feature extraction với Transformer
   encoder, cải thiện 2.3% so với pure CNN. [Paper C, Section 4, p.12]
```

### Đối tượng sử dụng

| Đối tượng | Nhu cầu |
|-----------|---------|
| Nghiên cứu sinh | Literature review, tìm research gap |
| Kỹ sư ML/AI | So sánh methods, tìm SOTA |
| Giảng viên | Tổng hợp tài liệu giảng dạy |
| Sinh viên | Đọc hiểu papers nhanh hơn |

---

## 2. Vấn đề cần giải quyết

### Tại sao cần dự án này?

```
❌ Vấn đề hiện tại:
┌──────────────────────────────────────────────────────────┐
│ • Đọc 50-100 papers mất hàng tuần                       │
│ • Không nhớ hết nội dung đã đọc                         │
│ • Khó so sánh kết quả giữa nhiều papers                 │
│ • Copy-paste thủ công vào bảng Excel                    │
│ • ChatGPT/Gemini trả lời chung chung, không có nguồn    │
│ • Ctrl+F không tìm được theo ngữ nghĩa                  │
└──────────────────────────────────────────────────────────┘

✅ Giải pháp của hệ thống:
┌──────────────────────────────────────────────────────────┐
│ • Upload papers → hệ thống tự đọc và index              │
│ • Hỏi bất kỳ câu hỏi nào → trả lời + trích dẫn        │
│ • Tự động tạo bảng so sánh methods/datasets/results     │
│ • Tìm kiếm theo ngữ nghĩa (semantic search)            │
│ • Lọc theo năm, tác giả, lĩnh vực                      │
│ • Conversation memory — hỏi tiếp theo ngữ cảnh         │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Kiến trúc hệ thống

### Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js)                           │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────┐ │
│  │ Upload   │  │ Chat         │  │ Paper      │  │ Comparison   │ │
│  │ Papers   │  │ Interface    │  │ Manager    │  │ Table View   │ │
│  └────┬─────┘  └──────┬───────┘  └─────┬──────┘  └──────┬───────┘ │
└───────┼────────────────┼────────────────┼────────────────┼─────────┘
        │                │                │                │
        ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     API GATEWAY (FastAPI)                            │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────┐ │
│  │ /upload  │  │ /chat        │  │ /papers    │  │ /compare     │ │
│  └────┬─────┘  └──────┬───────┘  └─────┬──────┘  └──────┬───────┘ │
└───────┼────────────────┼────────────────┼────────────────┼─────────┘
        │                │                │                │
        ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SERVICE LAYER                                  │
│                                                                     │
│  ┌─────────────────┐    ┌──────────────────────────────────────┐   │
│  │ INGESTION       │    │ RETRIEVAL + GENERATION               │   │
│  │ PIPELINE        │    │ PIPELINE                             │   │
│  │                 │    │                                      │   │
│  │ PDF Parser      │    │ Query Processor                      │   │
│  │     ▼           │    │     ▼                                │   │
│  │ Table Extractor │    │ Query Expansion (LLM)                │   │
│  │     ▼           │    │     ▼                                │   │
│  │ Semantic Chunker│    │ Hybrid Search                        │   │
│  │     ▼           │    │   ├─ BM25 (Sparse)                  │   │
│  │ Embedder (BGE)  │    │   └─ Dense Retrieval (Qdrant)       │   │
│  │     ▼           │    │     ▼                                │   │
│  │ Store           │    │ Reciprocal Rank Fusion               │   │
│  │                 │    │     ▼                                │   │
│  └─────────────────┘    │ Cross-Encoder Reranking              │   │
│                         │     ▼                                │   │
│                         │ LLM Generation (Gemini)              │   │
│                         │     ▼                                │   │
│                         │ Citation Grounding                   │   │
│                         │     ▼                                │   │
│                         │ Faithfulness Check                   │   │
│                         └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
        │                                │
        ▼                                ▼
┌───────────────┐    ┌───────────────────────────────┐
│   Qdrant      │    │   PostgreSQL                  │
│   (Vectors)   │    │   (Metadata, Users, History)  │
└───────────────┘    └───────────────────────────────┘
```

### Data Flow — Luồng dữ liệu

```
INGESTION (Upload paper mới):
  PDF File
    → Extract text + tables + figures metadata
    → Split thành chunks (500-1000 tokens, overlap 100)
    → Gắn metadata: {paper_id, section, page, authors, year}
    → Embed bằng BGE-M3 → vector 1024 chiều
    → Lưu vector vào Qdrant + metadata vào PostgreSQL

RETRIEVAL (Người dùng hỏi):
  User Query
    → Query Expansion: sinh 3-5 câu hỏi phụ
    → Hybrid Search: BM25 + Dense trên Qdrant
    → Reciprocal Rank Fusion: merge kết quả
    → Metadata Filter: lọc theo năm/tác giả nếu cần
    → Cross-Encoder Rerank: xếp hạng lại top-20 → top-5
    → LLM sinh câu trả lời + citations
    → Verify citations → trả về user
```

---

## 4. Tech Stack chi tiết

### Tại sao chọn từng công nghệ?

| Thành phần | Công nghệ | Lý do chọn |
|------------|-----------|-------------|
| **LLM** | Gemini API | Hỗ trợ context dài (1M tokens), giá rẻ, chất lượng tốt |
| **Embedding** | BGE-M3 | Hỗ trợ đa ngôn ngữ (EN/VI), sinh cả dense + sparse vectors |
| **Reranker** | bge-reranker-v2-m3 | Cross-encoder chất lượng cao, đa ngôn ngữ |
| **Vector DB** | Qdrant | Hỗ trợ hybrid search built-in, filtering mạnh, dễ deploy |
| **Backend** | FastAPI | Async, tự sinh docs, type-safe, hiệu suất cao |
| **Database** | PostgreSQL | Mature, JSONB cho metadata linh hoạt, full-text search |
| **Frontend** | Next.js 14 | SSR, App Router, React Server Components |
| **UI Library** | shadcn/ui + Tailwind | Components đẹp, customizable, lightweight |
| **PDF Parser** | PyMuPDF + Unstructured | PyMuPDF nhanh, Unstructured xử lý bảng tốt |
| **Deployment** | Docker Compose | Đóng gói tất cả services, dễ deploy lên VPS |

### So sánh các lựa chọn thay thế

#### Vector Database

| Tiêu chí | Qdrant | Pinecone | Weaviate | ChromaDB |
|----------|--------|----------|----------|----------|
| Self-hosted | ✅ | ❌ (Cloud only) | ✅ | ✅ |
| Hybrid Search | ✅ Built-in | ✅ | ✅ | ❌ |
| Filtering | ✅ Mạnh | ✅ | ✅ | ⚠️ Cơ bản |
| Production-ready | ✅ | ✅ | ✅ | ❌ |
| Dễ setup | ✅ | ✅ | ⚠️ | ✅ |
| **Kết luận** | **Chọn** | Tốt nhưng trả phí | Phức tạp hơn | Chỉ phù hợp prototype |

#### Embedding Model

| Tiêu chí | BGE-M3 | OpenAI Ada-002 | E5-Large | Cohere |
|----------|--------|----------------|----------|--------|
| Đa ngôn ngữ | ✅ Tốt | ✅ | ⚠️ | ✅ |
| Dense + Sparse | ✅ | ❌ | ❌ | ❌ |
| Self-hosted | ✅ | ❌ | ✅ | ❌ |
| Chi phí | Miễn phí | \$0.0001/1K tokens | Miễn phí | Trả phí |
| Chất lượng | ✅ Top-tier | ✅ | ✅ | ✅ |
| **Kết luận** | **Chọn** | Tốt nhưng trả phí | Không có sparse | Trả phí |

---

## 5. RAG Pipeline — Giải thích từng bước

### 5.1 Ingestion Pipeline (Nạp dữ liệu)

#### Bước 1: PDF Parsing

```
Mục tiêu: Trích xuất text, bảng, metadata từ file PDF

Thách thức:
  • Paper khoa học thường có layout 2 cột
  • Bảng kết quả phức tạp (merged cells, multi-row headers)
  • Công thức toán (LaTeX)
  • Header/Footer/Page numbers cần loại bỏ
  • Hình ảnh có caption quan trọng

Giải pháp:
  • PyMuPDF: Extract text nhanh, giữ layout
  • Unstructured: Xử lý bảng, nhận diện sections
  • Regex: Trích xuất metadata (title, authors, abstract)
```

```python
# Ví dụ cơ bản: PDF Parsing
import fitz  # PyMuPDF

def parse_pdf(file_path: str) -> dict:
    doc = fitz.open(file_path)
    result = {
        "metadata": extract_metadata(doc),
        "sections": [],
        "tables": [],
    }

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        # Xử lý layout 2 cột
        blocks = page.get_text("dict")["blocks"]
        # Phân loại blocks thành text/table/image
        for block in blocks:
            if block["type"] == 0:  # Text
                result["sections"].append({
                    "text": extract_text(block),
                    "page": page_num + 1,
                    "bbox": block["bbox"],
                })
            elif block["type"] == 1:  # Image
                # Lưu image + caption
                pass

    return result
```

#### Bước 2: Semantic Chunking

```
Mục tiêu: Chia document thành các đoạn có nghĩa, phù hợp cho retrieval

Tại sao KHÔNG dùng Fixed-size chunking?
  • Cắt ngang câu → mất nghĩa
  • Không biết chunk thuộc section nào
  • Không tối ưu cho semantic search

Chiến lược Semantic Chunking:
  1. Chia theo section (Abstract, Introduction, Method, Results, Conclusion)
  2. Trong mỗi section, chia theo paragraph
  3. Nếu paragraph quá dài (>1000 tokens), chia tiếp theo câu
  4. Overlap 10-20% giữa các chunks liên tiếp
  5. Mỗi chunk gắn metadata đầy đủ
```

```
Ví dụ một chunk tốt:

┌──────────────────────────────────────────────────────┐
│ Chunk ID: paper_001_method_chunk_03                  │
│ Paper: "Plant Disease Detection using Deep Learning" │
│ Section: Methodology                                 │
│ Page: 5-6                                            │
│ Authors: ["Nguyen A", "Tran B"]                      │
│ Year: 2024                                           │
│ ─────────────────────────────────────────────────     │
│ Text:                                                │
│ "We employed a ResNet-50 architecture pretrained     │
│  on ImageNet as the backbone. The model was          │
│  fine-tuned on PlantVillage dataset containing       │
│  54,306 images across 38 disease categories.         │
│  We used SGD optimizer with learning rate 0.001      │
│  and batch size 32. Data augmentation included       │
│  random rotation, horizontal flip, and color         │
│  jittering..."                                       │
└──────────────────────────────────────────────────────┘
```

#### Bước 3: Embedding

```
Mục tiêu: Chuyển text thành vector số để tìm kiếm theo ngữ nghĩa

BGE-M3 tạo ra 2 loại vector cùng lúc:
  • Dense vector (1024 chiều): Nắm bắt ngữ nghĩa sâu
  • Sparse vector: Giống BM25, tốt cho keyword matching

Tại sao cần cả 2?
  • Dense: "methods for detecting plant diseases" ≈ "approaches to identify crop pathology"
  • Sparse: Tìm chính xác từ khóa "ResNet-50", "PlantVillage"
  • Kết hợp → Hybrid Search mạnh hơn từng loại riêng lẻ
```

```python
# Ví dụ: Embedding với BGE-M3
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

# Embed một chunk
chunk_text = "We employed a ResNet-50 architecture..."
output = model.encode(
    [chunk_text],
    return_dense=True,
    return_sparse=True,
)

dense_vector = output["dense_vecs"][0]   # shape: (1024,)
sparse_vector = output["lexical_weights"][0]  # dict: {token_id: weight}
```

#### Bước 4: Lưu trữ vào Qdrant

```python
# Ví dụ: Upsert vào Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

client = QdrantClient(host="localhost", port=6333)

# Mỗi chunk = 1 point trong Qdrant
point = PointStruct(
    id=chunk_id,
    vector={
        "dense": dense_vector.tolist(),
        "sparse": SparseVector(
            indices=list(sparse_vector.keys()),
            values=list(sparse_vector.values()),
        ),
    },
    payload={
        "paper_id": "paper_001",
        "title": "Plant Disease Detection using Deep Learning",
        "section": "methodology",
        "page": 5,
        "authors": ["Nguyen A", "Tran B"],
        "year": 2024,
        "venue": "CVPR",
        "text": chunk_text,
    },
)

client.upsert(collection_name="papers", points=[point])
```

---

### 5.2 Retrieval Pipeline (Truy xuất)

#### Tổng quan Retrieval Pipeline

```
User Query: "ResNet và EfficientNet khác nhau như thế nào?"
                          │
                          ▼
              ┌───────────────────────┐
              │   Query Processing    │
              │                       │
              │ 1. Phân tích intent   │
              │ 2. Trích xuất filters │
              │    (năm, tác giả...) │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Query Expansion     │
              │                       │
              │ Sinh thêm queries:    │
              │ • "ResNet architecture │
              │    vs EfficientNet"   │
              │ • "comparison CNN     │
              │    backbones"         │
              │ • "accuracy ResNet    │
              │    EfficientNet       │
              │    benchmark"         │
              └───────────┬───────────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
        ┌──────────────┐   ┌──────────────┐
        │  BM25 Search │   │ Dense Search │
        │  (Keyword)   │   │ (Semantic)   │
        │              │   │              │
        │ Matches:     │   │ Matches:     │
        │ "ResNet"     │   │ ~ "so sánh   │
        │ "EfficientNet│   │   kiến trúc  │
        │              │   │   CNN"        │
        └──────┬───────┘   └──────┬───────┘
               │                  │
               └────────┬─────────┘
                        ▼
              ┌───────────────────────┐
              │ Reciprocal Rank       │
              │ Fusion (RRF)          │
              │                       │
              │ Merge & deduplicate   │
              │ top results from      │
              │ both searches         │
              │ → Top 20 candidates   │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Cross-Encoder         │
              │ Reranking             │
              │                       │
              │ Score mỗi pair        │
              │ (query, chunk)        │
              │ → Chọn Top 5          │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Top 5 chunks + meta   │
              │ → Gửi cho LLM        │
              └───────────────────────┘
```

#### Hybrid Search = BM25 + Dense

```
Tại sao cần Hybrid Search?

Ví dụ query: "EfficientNet-B7 accuracy on ImageNet"

Dense Search (semantic):
  ✅ Tìm được: "The model achieved 84.3% top-1 accuracy on the 
     ImageNet validation set" (không chứa từ "EfficientNet-B7")
  ❌ Có thể trả về kết quả về ResNet nếu ngữ cảnh tương tự

BM25 (keyword):
  ✅ Tìm chính xác: đoạn chứa "EfficientNet-B7" và "ImageNet"
  ❌ Bỏ sót: đoạn nói về cùng model nhưng dùng từ khác
     (ví dụ: "the largest variant of the Efficient family")

Hybrid = Tốt nhất của cả hai!
```

**Reciprocal Rank Fusion (RRF):**

```
Công thức: RRF_score(d) = Σ  1 / (k + rank_i(d))

Trong đó:
  • d = document (chunk)
  • k = constant (thường = 60)
  • rank_i(d) = thứ hạng của d trong hệ thống search thứ i

Ví dụ:
  Chunk X: rank 3 trong BM25, rank 7 trong Dense
  RRF(X) = 1/(60+3) + 1/(60+7) = 0.0159 + 0.0149 = 0.0308

  Chunk Y: rank 10 trong BM25, rank 1 trong Dense
  RRF(Y) = 1/(60+10) + 1/(60+1) = 0.0143 + 0.0164 = 0.0307

  → Chunk X và Y có score gần nhau, cả 2 đều được giữ lại
```

#### Cross-Encoder Reranking

```
Tại sao cần Reranking?

Bi-Encoder (embedding search):
  • Encode query và document RIÊNG RẼ → so sánh vector
  • Nhanh (milliseconds cho triệu documents)
  • Nhưng không nắm bắt được interaction giữa query-document

Cross-Encoder (reranking):
  • Encode query VÀ document CÙNG LÚC → attention giữa 2 bên
  • Chính xác hơn nhiều
  • Nhưng chậm → chỉ dùng cho top 20-50 candidates

Pipeline: Bi-Encoder (triệu → 20) → Cross-Encoder (20 → 5)
```

```python
# Ví dụ: Cross-Encoder Reranking
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

query = "ResNet và EfficientNet khác nhau như thế nào?"
candidates = [chunk.text for chunk in top_20_chunks]

# Score mỗi cặp (query, candidate)
pairs = [[query, candidate] for candidate in candidates]
scores = reranker.predict(pairs)

# Sắp xếp theo score giảm dần, lấy top 5
ranked = sorted(zip(scores, top_20_chunks), reverse=True)
top_5 = [chunk for _, chunk in ranked[:5]]
```

---

### 5.3 Generation Pipeline (Sinh câu trả lời)

#### Prompt Engineering cho Citation

```python
SYSTEM_PROMPT = """
Bạn là trợ lý nghiên cứu khoa học. Nhiệm vụ:
1. Trả lời câu hỏi DỰA TRÊN CÁC ĐOẠN TRÍCH được cung cấp
2. KHÔNG tự bịa thông tin ngoài context
3. Trích dẫn nguồn cho MỖI nhận định theo format: [Paper Title, Section, Page X]
4. Nếu context không đủ để trả lời, nói rõ "Dựa trên các papers hiện có, 
   tôi không tìm thấy thông tin về..."
5. Khi so sánh, tạo bảng markdown rõ ràng

Format trích dẫn:
  • Nhận định đơn: "ResNet đạt 95.2% accuracy [Paper A, Results, p.8]"
  • Nhận định tổng hợp: "Cả 3 nghiên cứu đều sử dụng PlantVillage dataset 
    [Paper A, p.3; Paper B, p.5; Paper C, p.4]"
"""

USER_PROMPT_TEMPLATE = """
## Context (các đoạn trích từ papers):

{formatted_chunks_with_metadata}

## Câu hỏi: {user_query}

## Yêu cầu: Trả lời chi tiết, kèm trích dẫn cho mỗi nhận định.
"""
```

#### Citation Grounding & Faithfulness Check

```
Sau khi LLM sinh câu trả lời, cần verify:

1. Citation Grounding:
   • Parse tất cả citations trong câu trả lời
   • Map mỗi citation về chunk nguồn cụ thể
   • Kiểm tra claim có thực sự xuất hiện trong chunk không

2. Faithfulness Check (dùng LLM-as-judge):
   • Cho LLM thứ 2 đánh giá: "Câu trả lời này có được hỗ trợ bởi context không?"
   • Score: Faithful / Partially Faithful / Not Faithful
   • Nếu Not Faithful → regenerate hoặc cảnh báo user

3. Hallucination Detection:
   • So sánh entities/numbers trong câu trả lời vs context
   • Flag nếu xuất hiện thông tin không có trong context
```

---

## 6. Các kỹ thuật nâng cao

### 6.1 Multi-Query Retrieval

```
Vấn đề: Một câu hỏi có thể được diễn đạt nhiều cách
  → Một query duy nhất có thể miss relevant chunks

Giải pháp: Dùng LLM sinh 3-5 câu hỏi phụ, retrieve riêng, merge kết quả

Ví dụ:
  Original: "Hạn chế của Vision Transformer trong medical imaging?"
  
  Generated queries:
  1. "Limitations of ViT for medical image analysis"
  2. "Challenges applying transformer to healthcare images"
  3. "Drawbacks of self-attention mechanism in clinical imaging"
  4. "Vision Transformer computational cost medical datasets"
  
  → Retrieve cho mỗi query → merge → deduplicate → rerank
```

### 6.2 So sánh nhiều Papers (Comparison Table)

```
Đây là tính năng nổi bật nhất cho portfolio!

User: "So sánh phương pháp, dataset, metric và kết quả của 5 bài báo"

Hệ thống:
  1. Retrieve chunks từ section "Method" và "Results" của mỗi paper
  2. Extract structured info: method, dataset, metrics, results
  3. LLM tổng hợp thành bảng:

  | Paper | Method | Dataset | Metric | Result |
  |-------|--------|---------|--------|--------|
  | Paper A | ResNet-50 | PlantVillage | Accuracy | 95.2% |
  | Paper B | EfficientNet-B4 | PlantVillage | Accuracy | 96.8% |
  | Paper C | ViT-B/16 | PlantVillage | Accuracy | 97.1% |
  | Paper D | ResNet + ViT | Custom | F1-Score | 94.5% |
  | Paper E | MobileNet-V3 | PlantDoc | Accuracy | 91.3% |
  
  4. Mỗi ô trong bảng có citation link về nguồn gốc
```

### 6.3 Metadata Filtering

```
Cho phép user thu hẹp phạm vi search bằng metadata:

Filters có thể dùng:
  • Năm: year >= 2022
  • Tác giả: authors contains "Nguyen"
  • Venue: venue in ["CVPR", "ICCV", "NeurIPS"]
  • Lĩnh vực: field = "Computer Vision"
  • Section: section = "methodology"

Cách hoạt động trong Qdrant:
  1. User hỏi: "Các phương pháp transfer learning năm 2023-2024?"
  2. System detect filter: year in [2023, 2024]
  3. Qdrant search với filter:
     must: [{key: "year", range: {gte: 2023, lte: 2024}}]
  4. Search chỉ trong subset papers phù hợp
```

---

## 7. Database Schema

### PostgreSQL Schema

```sql
-- Bảng papers: Lưu metadata của mỗi bài báo
CREATE TABLE papers (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title         TEXT NOT NULL,
    authors       TEXT[] NOT NULL,           -- Array of author names
    year          INTEGER,
    venue         VARCHAR(255),              -- CVPR, NeurIPS, etc.
    abstract      TEXT,
    field         VARCHAR(100),              -- Computer Vision, NLP, etc.
    file_path     VARCHAR(500) NOT NULL,     -- Path to stored PDF
    file_hash     VARCHAR(64) UNIQUE,        -- SHA-256 to prevent duplicates
    total_pages   INTEGER,
    status        VARCHAR(20) DEFAULT 'processing',  -- processing/ready/error
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

-- Bảng chunks: Lưu thông tin từng chunk (text lưu cả ở Qdrant payload)
CREATE TABLE chunks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id      UUID REFERENCES papers(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,          -- Thứ tự chunk trong paper
    section       VARCHAR(100),              -- abstract, introduction, method...
    page_start    INTEGER,
    page_end      INTEGER,
    text          TEXT NOT NULL,
    token_count   INTEGER,
    qdrant_id     UUID,                      -- ID tương ứng trong Qdrant
    created_at    TIMESTAMP DEFAULT NOW()
);

-- Bảng conversations: Lưu lịch sử hội thoại
CREATE TABLE conversations (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id),
    title         VARCHAR(500),
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

-- Bảng messages: Lưu từng tin nhắn trong conversation
CREATE TABLE messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role            VARCHAR(20) NOT NULL,    -- 'user' or 'assistant'
    content         TEXT NOT NULL,
    citations       JSONB,                   -- Array of citation objects
    chunks_used     UUID[],                  -- IDs of chunks used for this answer
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Bảng users (cơ bản)
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    name          VARCHAR(255),
    created_at    TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_papers_year ON papers(year);
CREATE INDEX idx_papers_field ON papers(field);
CREATE INDEX idx_chunks_paper_id ON chunks(paper_id);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_papers_authors ON papers USING GIN(authors);
```

### Qdrant Collection Schema

```json
{
  "collection_name": "papers",
  "vectors": {
    "dense": {
      "size": 1024,
      "distance": "Cosine"
    }
  },
  "sparse_vectors": {
    "sparse": {
      "modifier": "idf"
    }
  },
  "payload_schema": {
    "paper_id": "uuid",
    "title": "keyword",
    "section": "keyword",
    "page": "integer",
    "year": "integer",
    "authors": "keyword[]",
    "venue": "keyword",
    "field": "keyword",
    "text": "text"
  }
}
```

---

## 8. API Endpoints

### Papers Management

```
POST   /api/v1/papers/upload          Upload một hoặc nhiều PDF
GET    /api/v1/papers                  Danh sách tất cả papers
GET    /api/v1/papers/{id}             Chi tiết một paper
DELETE /api/v1/papers/{id}             Xóa paper và chunks liên quan
GET    /api/v1/papers/{id}/chunks      Xem các chunks của paper
GET    /api/v1/papers/{id}/status      Kiểm tra trạng thái processing
```

### Chat & Query

```
POST   /api/v1/chat                    Gửi câu hỏi, nhận câu trả lời + citations
POST   /api/v1/chat/stream             Streaming response (SSE)
GET    /api/v1/conversations            Danh sách conversations
GET    /api/v1/conversations/{id}       Chi tiết conversation + messages
DELETE /api/v1/conversations/{id}       Xóa conversation
```

### Comparison & Analysis

```
POST   /api/v1/compare                 So sánh nhiều papers → bảng tổng hợp
POST   /api/v1/search                  Tìm kiếm papers với filters
GET    /api/v1/stats                   Thống kê: số papers, chunks, queries
```

### Ví dụ Request/Response

```json
// POST /api/v1/chat
// Request:
{
  "conversation_id": "uuid-or-null-for-new",
  "query": "ResNet và EfficientNet khác nhau như thế nào?",
  "filters": {
    "year_min": 2020,
    "fields": ["Computer Vision"]
  },
  "paper_ids": ["uuid-1", "uuid-2"],  // optional: giới hạn search trong papers cụ thể
  "top_k": 5
}

// Response:
{
  "conversation_id": "abc-123",
  "answer": "Dựa trên các nghiên cứu đã cung cấp, ResNet và EfficientNet có những khác biệt chính sau:\n\n1. **Kiến trúc**: ResNet sử dụng skip connections...[Paper A, Section 2, p.3]\n\n2. **Hiệu suất**: EfficientNet đạt accuracy cao hơn 1.6%...[Paper B, Table 1, p.7]",
  "citations": [
    {
      "paper_id": "uuid-1",
      "paper_title": "Deep Residual Learning for Image Recognition",
      "section": "Architecture",
      "page": 3,
      "chunk_text": "The core idea of ResNet is the introduction of skip connections...",
      "relevance_score": 0.94
    },
    {
      "paper_id": "uuid-2",
      "paper_title": "EfficientNet: Rethinking Model Scaling",
      "section": "Results",
      "page": 7,
      "chunk_text": "Our EfficientNet-B7 achieves 84.3% top-1 accuracy...",
      "relevance_score": 0.91
    }
  ],
  "metadata": {
    "total_chunks_searched": 1250,
    "chunks_retrieved": 20,
    "chunks_after_rerank": 5,
    "latency_ms": 2340
  }
}
```

---

## 9. Cấu trúc thư mục

```
ai-research-assistant/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app entry point
│   │   ├── config.py                  # Settings & environment variables
│   │   │
│   │   ├── api/                       # API Routes
│   │   │   ├── __init__.py
│   │   │   ├── papers.py              # Paper upload, list, delete
│   │   │   ├── chat.py                # Chat & query endpoints
│   │   │   ├── compare.py             # Paper comparison
│   │   │   └── search.py              # Search with filters
│   │   │
│   │   ├── models/                    # SQLAlchemy ORM Models
│   │   │   ├── __init__.py
│   │   │   ├── paper.py
│   │   │   ├── chunk.py
│   │   │   ├── conversation.py
│   │   │   └── message.py
│   │   │
│   │   ├── schemas/                   # Pydantic Schemas (request/response)
│   │   │   ├── __init__.py
│   │   │   ├── paper.py
│   │   │   ├── chat.py
│   │   │   └── compare.py
│   │   │
│   │   ├── services/                  # Business Logic
│   │   │   ├── __init__.py
│   │   │   ├── pdf_parser.py          # PDF text & table extraction
│   │   │   ├── chunker.py             # Semantic chunking logic
│   │   │   ├── embedder.py            # BGE-M3 embedding
│   │   │   ├── retriever.py           # Hybrid search + RRF
│   │   │   ├── reranker.py            # Cross-encoder reranking
│   │   │   ├── generator.py           # LLM generation + citation
│   │   │   ├── query_processor.py     # Query expansion, multi-query
│   │   │   └── evaluator.py           # RAG evaluation metrics
│   │   │
│   │   ├── db/                        # Database
│   │   │   ├── __init__.py
│   │   │   ├── session.py             # DB session management
│   │   │   ├── qdrant.py              # Qdrant client
│   │   │   └── migrations/            # Alembic migrations
│   │   │
│   │   └── utils/                     # Helpers
│   │       ├── __init__.py
│   │       ├── text_processing.py
│   │       └── citation_parser.py
│   │
│   ├── tests/                         # Unit & integration tests
│   │   ├── test_pdf_parser.py
│   │   ├── test_chunker.py
│   │   ├── test_retriever.py
│   │   └── test_generator.py
│   │
│   ├── scripts/                       # Utility scripts
│   │   ├── seed_papers.py             # Load sample papers
│   │   └── evaluate_rag.py            # Run RAG evaluation
│   │
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── app/                       # Next.js App Router
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx               # Landing / Dashboard
│   │   │   ├── chat/
│   │   │   │   └── page.tsx           # Chat interface
│   │   │   ├── papers/
│   │   │   │   └── page.tsx           # Paper management
│   │   │   └── compare/
│   │   │       └── page.tsx           # Comparison view
│   │   │
│   │   ├── components/
│   │   │   ├── ui/                    # shadcn/ui components
│   │   │   ├── ChatMessage.tsx        # Message bubble with citations
│   │   │   ├── CitationPanel.tsx      # Side panel showing source
│   │   │   ├── PaperUploader.tsx      # Drag-drop upload
│   │   │   ├── PaperList.tsx          # Paper management table
│   │   │   ├── ComparisonTable.tsx    # Paper comparison grid
│   │   │   └── FilterBar.tsx          # Year, author, field filters
│   │   │
│   │   └── lib/
│   │       ├── api.ts                 # API client
│   │       └── types.ts               # TypeScript types
│   │
│   ├── package.json
│   ├── Dockerfile
│   └── tailwind.config.ts
│
├── docker-compose.yml                 # Orchestrate all services
├── .env.example                       # Environment variables template
├── Makefile                           # Common commands
└── README.md                          # Project documentation
```

---

## 10. Kế hoạch triển khai theo Phase

### Tổng quan Timeline

```
Phase 1 (Tuần 1-2)    ████████░░░░░░░░░░░░  Foundation
Phase 2 (Tuần 3-4)    ░░░░░░░░████████░░░░  Retrieval Pipeline
Phase 3 (Tuần 5-6)    ░░░░░░░░░░░░░░░░████  Generation + Citation
Phase 4 (Tuần 7-8)    ░░░░░░░░░░░░░░░░████  Frontend
Phase 5 (Tuần 9-10)   ░░░░░░░░░░░░░░░░████  Evaluation + Polish
```

### Phase 1: Foundation (Tuần 1-2)

```
Mục tiêu: Upload PDF → Parse → Chunk → Embed → Store

Tuần 1:
  ☐ Setup project structure
  ☐ Docker Compose: FastAPI + PostgreSQL + Qdrant
  ☐ Database models + migrations (Alembic)
  ☐ PDF Parser: text extraction + metadata
  ☐ Basic upload endpoint

Tuần 2:
  ☐ Table extraction từ PDF
  ☐ Semantic Chunking implementation
  ☐ BGE-M3 embedding integration
  ☐ Qdrant collection setup + upsert
  ☐ Paper management endpoints (CRUD)

Deliverable: Upload PDF → tự parse, chunk, embed, store ✅
```

### Phase 2: Retrieval Pipeline (Tuần 3-4)

```
Mục tiêu: Query → Search → Rerank → Return relevant chunks

Tuần 3:
  ☐ Dense retrieval qua Qdrant
  ☐ BM25 sparse search
  ☐ Hybrid search + RRF fusion
  ☐ Metadata filtering

Tuần 4:
  ☐ Cross-Encoder reranking
  ☐ Query expansion (LLM-based)
  ☐ Multi-query retrieval
  ☐ Search endpoint + testing

Deliverable: Hỏi câu hỏi → trả về top-5 relevant chunks ✅
```

### Phase 3: Generation + Citation (Tuần 5-6)

```
Mục tiêu: LLM trả lời có trích dẫn chính xác

Tuần 5:
  ☐ Gemini API integration
  ☐ Prompt engineering cho citation
  ☐ Citation parsing & grounding
  ☐ Chat endpoint (single-turn)

Tuần 6:
  ☐ Conversation memory (multi-turn)
  ☐ Streaming response (SSE)
  ☐ Faithfulness check
  ☐ Paper comparison endpoint
  ☐ Comparison table generation

Deliverable: Full Q&A với citations + so sánh papers ✅
```

### Phase 4: Frontend (Tuần 7-8)

```
Mục tiêu: Giao diện người dùng hoàn chỉnh

Tuần 7:
  ☐ Next.js project setup
  ☐ Chat interface (message list, input)
  ☐ Citation panel (click citation → show source)
  ☐ Paper upload (drag-drop)

Tuần 8:
  ☐ Paper management page
  ☐ Filter bar (year, author, field)
  ☐ Comparison table view
  ☐ Conversation history sidebar
  ☐ Responsive design

Deliverable: Giao diện hoàn chỉnh, kết nối API ✅
```

### Phase 5: Evaluation + Polish (Tuần 9-10)

```
Mục tiêu: Đo lường chất lượng + deploy

Tuần 9:
  ☐ Tạo evaluation dataset (50-100 Q&A pairs)
  ☐ Implement metrics: Recall@K, MRR, nDCG
  ☐ Faithfulness evaluation (LLM-as-judge)
  ☐ A/B test: naive RAG vs full pipeline
  ☐ Benchmark report

Tuần 10:
  ☐ Performance optimization
  ☐ Error handling & edge cases
  ☐ Dockerize all services
  ☐ Deploy lên VPS
  ☐ Write README + demo video

Deliverable: Production-ready system + evaluation report ✅
```

---

## 11. Đánh giá chất lượng (RAG Evaluation)

### Tại sao cần đánh giá?

```
Không đánh giá = không biết hệ thống hoạt động tốt hay tệ
→ Đây là điểm mạnh nhất để phân biệt project của bạn vs project demo

Evaluation gồm 2 phần:
  1. Retrieval Quality: Tìm đúng chunks không?
  2. Generation Quality: Trả lời đúng và có citations chính xác không?
```

### Retrieval Metrics

| Metric | Ý nghĩa | Cách tính |
|--------|---------|-----------|
| **Recall@K** | Trong top-K results, bao nhiêu % là relevant? | `relevant_in_topK / total_relevant` |
| **Precision@K** | Top-K results có bao nhiêu % là relevant? | `relevant_in_topK / K` |
| **MRR** | Result relevant đầu tiên ở vị trí nào? | `1 / rank_of_first_relevant` |
| **nDCG** | Xếp hạng có tốt không? (relevant docs có ở top không?) | Xem công thức bên dưới |

```
Ví dụ tính MRR:
  Query 1: relevant doc ở vị trí 1 → RR = 1/1 = 1.0
  Query 2: relevant doc ở vị trí 3 → RR = 1/3 = 0.33
  Query 3: relevant doc ở vị trí 2 → RR = 1/2 = 0.5

  MRR = (1.0 + 0.33 + 0.5) / 3 = 0.61
```

### Generation Metrics

| Metric | Ý nghĩa | Cách đánh giá |
|--------|---------|---------------|
| **Faithfulness** | Câu trả lời có đúng với context không? | LLM-as-judge: check từng claim |
| **Answer Relevancy** | Câu trả lời có liên quan câu hỏi không? | Cosine similarity query ↔ answer |
| **Citation Accuracy** | Citations có chính xác không? | Verify claim ↔ cited chunk |
| **Completeness** | Trả lời có đầy đủ không? | So sánh với ground-truth answer |

### So sánh Pipeline

```
Tạo bảng so sánh để chứng minh giá trị của từng component:

| Pipeline Config        | Recall@5 | MRR  | Faithfulness |
|------------------------|----------|------|--------------|
| Naive (dense only)     | 0.62     | 0.45 | 0.71         |
| + BM25 Hybrid          | 0.74     | 0.58 | 0.73         |
| + Reranking            | 0.81     | 0.72 | 0.75         |
| + Query Expansion      | 0.86     | 0.76 | 0.76         |
| + Better Prompting     | 0.86     | 0.76 | 0.89         |
| Full Pipeline          | 0.86     | 0.76 | 0.89         |

→ Mỗi component cải thiện bao nhiêu %? Rất thuyết phục cho portfolio!
```

---

## 12. Deployment

### Docker Compose

```yaml
# docker-compose.yml
version: "3.8"

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/research_db
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    depends_on:
      - postgres
      - qdrant
    volumes:
      - ./data/papers:/app/papers  # Stored PDFs

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000

  postgres:
    image: postgres:16
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=research_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  postgres_data:
  qdrant_data:
```

### VPS Requirements

```
Minimum (cho demo):
  • 4 vCPU
  • 8 GB RAM (BGE-M3 cần ~2GB, Reranker ~1GB)
  • 50 GB SSD
  • Ubuntu 22.04

Recommended (cho production):
  • 8 vCPU
  • 16 GB RAM
  • 100 GB SSD
  • GPU (optional, cho embedding nhanh hơn)
```

---

## 13. Rủi ro & Giải pháp

| Rủi ro | Mức độ | Giải pháp |
|--------|--------|-----------|
| PDF parsing lỗi layout 2 cột | Cao | Dùng Unstructured + fallback PyMuPDF, test nhiều papers |
| BGE-M3 chậm trên CPU | Trung bình | Batch embedding, cache results, dùng ONNX Runtime |
| LLM hallucinate citations | Cao | Faithfulness check, constrained generation, verify mỗi citation |
| Qdrant hết memory | Thấp | Scalar quantization, disk-based index |
| Context window overflow | Trung bình | Giới hạn top-K chunks, summarize nếu quá dài |
| Chi phí Gemini API | Thấp | Cache responses, dùng Flash model cho query expansion |
| Bảng trong PDF parse sai | Cao | Dùng Camelot/Tabula chuyên biệt, OCR fallback |

---

## 14. Tài liệu tham khảo

### Papers & Articles

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Paper gốc về RAG
- [BGE M3-Embedding](https://arxiv.org/abs/2402.03216) — Multi-lingual, multi-granularity embedding
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — Vấn đề LLM bỏ sót thông tin ở giữa context
- [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217) — Framework đánh giá RAG

### Documentation

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Unstructured Documentation](https://docs.unstructured.io/)

### Useful Tools

- [RAGAS](https://docs.ragas.io/) — RAG evaluation framework
- [LangSmith](https://smith.langchain.com/) — LLM observability & tracing
- [Sentence Transformers](https://www.sbert.net/) — Embedding & reranking models

---

> **Ghi chú cuối**: Document này là blueprint kỹ thuật. Khi bắt đầu code,
> hãy đi từ Phase 1 và test kỹ từng component trước khi ghép lại.
> Một hệ thống RAG tốt không phải do có nhiều tính năng, mà do mỗi bước
> trong pipeline đều hoạt động chính xác.
