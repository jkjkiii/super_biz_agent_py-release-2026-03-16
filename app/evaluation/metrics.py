import numpy as np
from typing import List, Set

def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """计算 Recall@K"""
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    hit = len(retrieved_set & relevant_ids)
    return hit / len(relevant_ids)

def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """计算 MRR (Mean Reciprocal Rank)"""
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved_ids: List[str], relevance_scores: dict, k: int) -> float:
    """
    计算 NDCG@K
    relevance_scores: {doc_id: score}，score 为分级相关性（如 0/1/2）
    """
    if not relevance_scores:
        return 0.0
    # 计算 DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = relevance_scores.get(doc_id, 0)
        dcg += rel / np.log2(i + 1)
    # 计算 IDCG（理想排序）
    sorted_relevances = sorted(relevance_scores.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(sorted_relevances[:k], start=1):
        idcg += rel / np.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0