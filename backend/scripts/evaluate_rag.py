import sys
import os
import uuid

# Configure stdout for Windows console UTF-8 compatibility
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure backend directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.qdrant import get_qdrant_client, init_qdrant_collection, upsert_chunks_to_qdrant
from app.services.embedder import Embedder
from app.services.retriever import HybridRetriever
from app.services.evaluator import RAGEvaluator

def make_uuid(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))

def run_rag_benchmark():
    print("=================================================================")
    print("[*] RUNNING RAG EVALUATION BENCHMARK (IR METRICS)")
    print("=================================================================\n")

    client = get_qdrant_client()
    init_qdrant_collection(client)
    embedder = Embedder()
    retriever = HybridRetriever(client=client, embedder=embedder)

    id_resnet = make_uuid("c_resnet_01")
    id_vit = make_uuid("c_vit_01")
    id_eff = make_uuid("c_eff_01")
    id_plant = make_uuid("c_plant_01")

    # 1. Seed Ground Truth Dataset (Papers, Chunks & Queries)
    test_corpus = [
        {
            "id": id_resnet,
            "paper_id": make_uuid("p_resnet"),
            "paper_title": "Deep Residual Learning for Image Recognition",
            "year": 2016,
            "section": "Architecture",
            "page_start": 3,
            "page_end": 4,
            "text": "ResNet utilizes identity shortcut connections to solve vanishing gradients in very deep networks.",
            "token_count": 20
        },
        {
            "id": id_vit,
            "paper_id": make_uuid("p_vit"),
            "paper_title": "An Image is Worth 16x16 Words",
            "year": 2020,
            "section": "Method",
            "page_start": 2,
            "page_end": 3,
            "text": "Vision Transformers split an image into fixed-size 16x16 patches and linearly embed each patch.",
            "token_count": 20
        },
        {
            "id": id_eff,
            "paper_id": make_uuid("p_eff"),
            "paper_title": "EfficientNet: Rethinking Model Scaling",
            "year": 2019,
            "section": "Scaling",
            "page_start": 4,
            "page_end": 5,
            "text": "Compound scaling uniformly scales network width, depth, and resolution with a fixed coefficient.",
            "token_count": 20
        },
        {
            "id": id_plant,
            "paper_id": make_uuid("p_plant"),
            "paper_title": "PlantVillage Crop Disease Dataset",
            "year": 2015,
            "section": "Dataset",
            "page_start": 1,
            "page_end": 2,
            "text": "The PlantVillage dataset contains 54,306 images of healthy and infected plant leaves across 38 classes.",
            "token_count": 22
        }
    ]

    print("Indexing corpus into Qdrant collection...")
    embeddings = embedder.embed_texts([c["text"] for c in test_corpus])
    upsert_chunks_to_qdrant(test_corpus, embeddings, client=client)

    # 2. Benchmark Queries and Ground Truth Relevant Chunk IDs
    benchmark_queries = [
        {
            "query": "How do identity shortcut connections solve vanishing gradients in ResNet?",
            "relevant_ids": [id_resnet]
        },
        {
            "query": "What patch size does Vision Transformer use for linear embedding?",
            "relevant_ids": [id_vit]
        },
        {
            "query": "Compound scaling coefficient depth width resolution",
            "relevant_ids": [id_eff]
        },
        {
            "query": "PlantVillage dataset image count categories crop pathology",
            "relevant_ids": [id_plant]
        }
    ]

    # Configurations to compare
    configs = [
        {"name": "1. Naive (Dense Only)", "apply_rerank": False, "dense_only": True},
        {"name": "2. Hybrid (Dense + BM25 RRF)", "apply_rerank": False, "dense_only": False},
        {"name": "3. Full Pipeline (+ Cross-Encoder Rerank)", "apply_rerank": True, "dense_only": False}
    ]

    results_table = []

    for cfg in configs:
        recalls = []
        mrrs = []
        ndcgs = []

        for item in benchmark_queries:
            q = item["query"]
            rel_ids = item["relevant_ids"]

            if cfg["dense_only"]:
                dense_emb = embedder.embed_query(q)["dense"]
                q_res = client.query_points(
                    collection_name="papers",
                    query=dense_emb,
                    using="dense",
                    limit=5
                )
                retrieved_ids = [str(pt.id) for pt in q_res.points]
            else:
                chunks = retriever.retrieve(
                    query=q,
                    limit=5,
                    candidates_k=10,
                    apply_rerank=cfg["apply_rerank"]
                )
                retrieved_ids = [c["id"] for c in chunks]

            recalls.append(RAGEvaluator.recall_at_k(rel_ids, retrieved_ids, k=5))
            mrrs.append(RAGEvaluator.mrr(rel_ids, retrieved_ids))
            ndcgs.append(RAGEvaluator.ndcg_at_k(rel_ids, retrieved_ids, k=5))

        avg_recall = sum(recalls) / len(recalls)
        avg_mrr = sum(mrrs) / len(mrrs)
        avg_ndcg = sum(ndcgs) / len(ndcgs)

        results_table.append({
            "config": cfg["name"],
            "recall": avg_recall,
            "mrr": avg_mrr,
            "ndcg": avg_ndcg
        })

    print("\n-----------------------------------------------------------------")
    print(f"{'Pipeline Configuration':<42} | {'Recall@5':<10} | {'MRR':<8} | {'nDCG@5':<8}")
    print("-----------------------------------------------------------------")
    for r in results_table:
        print(f"{r['config']:<42} | {r['recall']:<10.3f} | {r['mrr']:<8.3f} | {r['ndcg']:<8.3f}")
    print("-----------------------------------------------------------------\n")
    print("[+] Benchmark completed successfully!")

if __name__ == "__main__":
    run_rag_benchmark()
