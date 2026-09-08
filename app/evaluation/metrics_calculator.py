"""检索评估指标计算 - Recall@K, MRR, NDCG"""

from typing import List, Set, Dict
import math


def recall_at_k(
    retrieved_ids: List[str], 
    relevant_ids: Set[str], 
    k: int
) -> float:
    """
    计算 Recall@K
    
    Args:
        retrieved_ids: 检索返回的文档 ID 列表
        relevant_ids: 相关文档 ID 集合
        k: 截断点
        
    Returns:
        float: 召回率
    """
    if not relevant_ids:
        return 0.0
    
    retrieved_set = set(retrieved_ids[:k])
    hit = len(retrieved_set & relevant_ids)
    return hit / len(relevant_ids)


def precision_at_k(
    retrieved_ids: List[str], 
    relevant_ids: Set[str], 
    k: int
) -> float:
    """
    计算 Precision@K
    """
    if k == 0:
        return 0.0
    
    retrieved_set = set(retrieved_ids[:k])
    hit = len(retrieved_set & relevant_ids)
    return hit / k


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    计算 MRR (Mean Reciprocal Rank)
    
    返回第一个相关文档的倒数，如果没有则为 0
    """
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str], 
    relevance_scores: Dict[str, int], 
    k: int
) -> float:
    """
    计算 NDCG@K (Normalized Discounted Cumulative Gain)
    
    Args:
        retrieved_ids: 检索返回的文档 ID 列表
        relevance_scores: {doc_id: relevance_score}，分数为整数（0, 1, 2...）
        k: 截断点
        
    Returns:
        float: NDCG 值，范围 0~1
    """
    if not relevance_scores:
        return 0.0
    
    # 计算 DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = relevance_scores.get(doc_id, 0)
        dcg += rel / math.log2(i + 1)
    
    # 计算 IDCG（理想排序）
    sorted_relevances = sorted(relevance_scores.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(sorted_relevances[:k], start=1):
        idcg += rel / math.log2(i + 1)
    
    return dcg / idcg if idcg > 0 else 0.0


def map_score(
    retrieved_ids: List[str], 
    relevant_ids: Set[str]
) -> float:
    """
    计算 MAP (Mean Average Precision)
    """
    if not relevant_ids:
        return 0.0
    
    score = 0.0
    num_hits = 0
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            num_hits += 1
            score += num_hits / i
    
    return score / len(relevant_ids)


def all_metrics(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    relevance_scores: Dict[str, int],
    k_values: List[int] = [3, 5, 10]
) -> Dict[str, float]:
    """
    计算所有指标
    
    Returns:
        Dict: {
            'recall@3': ..., 'recall@5': ..., 'recall@10': ...,
            'precision@3': ..., 'precision@5': ..., 'precision@10': ...,
            'mrr': ..., 'ndcg@10': ..., 'map': ...
        }
    """
    metrics = {}
    
    for k in k_values:
        metrics[f'recall@{k}'] = recall_at_k(retrieved_ids, relevant_ids, k)
        metrics[f'precision@{k}'] = precision_at_k(retrieved_ids, relevant_ids, k)
    
    metrics['mrr'] = mrr(retrieved_ids, relevant_ids)
    if relevance_scores:
        metrics['ndcg@10'] = ndcg_at_k(retrieved_ids, relevance_scores, 10)
    else:
        # 如果没有分级分数，用二值相关计算 NDCG
        binary_scores = {doc_id: 1 for doc_id in relevant_ids}
        metrics['ndcg@10'] = ndcg_at_k(retrieved_ids, binary_scores, 10)
    
    metrics['map'] = map_score(retrieved_ids, relevant_ids)
    
    return metrics