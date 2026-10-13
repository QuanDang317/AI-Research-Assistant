import math
from typing import List, Dict, Any, Set


class RAGEvaluator:
    """
    RAG Evaluation Toolkit for academic retrieval and generation quality.
    Computes standard IR metrics: Recall@K, Precision@K, MRR, nDCG@K.
    """

    @staticmethod
    def recall_at_k(ground_truth_ids: List[str], retrieved_ids: List[str], k: int = 5) -> float:
        """
        Recall@K = |Retrieved relevant docs in top-K| / |Total relevant docs|
        """
        if not ground_truth_ids:
            return 1.0
        top_k = set(retrieved_ids[:k])
        relevant = set(ground_truth_ids)
        return len(top_k.intersection(relevant)) / float(len(relevant))

    @staticmethod
    def precision_at_k(ground_truth_ids: List[str], retrieved_ids: List[str], k: int = 5) -> float:
        """
        Precision@K = |Retrieved relevant docs in top-K| / K
        """
        if k <= 0:
            return 0.0
        top_k = set(retrieved_ids[:k])
        relevant = set(ground_truth_ids)
        return len(top_k.intersection(relevant)) / float(k)

    @staticmethod
    def mrr(ground_truth_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Mean Reciprocal Rank (MRR) = 1 / rank of first relevant doc.
        """
        relevant = set(ground_truth_ids)
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(ground_truth_ids: List[str], retrieved_ids: List[str], k: int = 5) -> float:
        """
        Normalized Discounted Cumulative Gain (nDCG@K)
        """
        top_k = retrieved_ids[:k]
        relevant = set(ground_truth_ids)

        dcg = 0.0
        for idx, doc_id in enumerate(top_k, start=1):
            rel = 1.0 if doc_id in relevant else 0.0
            dcg += (2.0 ** rel - 1.0) / math.log2(idx + 1)

        # Ideal DCG
        ideal_relevances = [1.0] * min(len(relevant), k)
        idcg = sum((2.0 ** rel - 1.0) / math.log2(idx + 1) for idx, rel in enumerate(ideal_relevances, start=1))

        if idcg == 0.0:
            return 0.0
        return dcg / idcg
