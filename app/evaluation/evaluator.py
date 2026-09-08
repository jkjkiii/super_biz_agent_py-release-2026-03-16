from typing import List, Dict, Any
from loguru import logger

from app.services.vector_store_manager import vector_store_manager
from app.rag.retrievers import hybrid_retriever
from app.evaluation.metrics_calculator import all_metrics as calculate_all_metrics


def _extract_doc_ids(docs) -> List[str]:
    """从 Document 列表中提取 ID"""
    ids = []
    for doc in docs:
        doc_id = doc.metadata.get("id")
        if doc_id:
            ids.append(doc_id)
    return ids


class RAGEvaluator:
    def __init__(self, test_data: List[Dict[str, Any]]):
        self.test_data = test_data
        self.results = []
    
    def evaluate_baseline(self, k_values: List[int] = [3, 5, 10]):
        """评估纯向量检索"""
        logger.info("开始 Baseline 评估 (纯向量检索)")
        
        all_metric_lists = {f"recall@{k}": [] for k in k_values}
        all_metric_lists["mrr"] = []
        all_metric_lists["ndcg@10"] = []
        all_metric_lists["map"] = []
        
        for item in self.test_data:
            query = item["query"]
            relevant_ids = set(item["relevant_ids"])
            relevance_scores = item.get("relevance_scores", {})
            
            # 使用最大 k 值一次性检索
            docs = vector_store_manager.similarity_search(query, k=max(k_values))
            retrieved_ids = _extract_doc_ids(docs)
            
            metrics = calculate_all_metrics(retrieved_ids, relevant_ids, relevance_scores, k_values)
            
            for k in k_values:
                all_metric_lists[f"recall@{k}"].append(metrics[f"recall@{k}"])
            all_metric_lists["mrr"].append(metrics["mrr"])
            all_metric_lists["ndcg@10"].append(metrics["ndcg@10"])
            all_metric_lists["map"].append(metrics["map"])
        
        avg_metrics = {k: sum(v)/len(v) if v else 0.0 for k, v in all_metric_lists.items()}
        logger.info(f"Baseline 结果: {avg_metrics}")
        return avg_metrics
    
    def evaluate_optimized(self, k_values: List[int] = [3, 5, 10]):
        """评估混合检索 + 重排序"""
        logger.info("开始 Optimized 评估 (混合检索 + 重排序)")
        
        all_metric_lists = {f"recall@{k}": [] for k in k_values}
        all_metric_lists["mrr"] = []
        all_metric_lists["ndcg@10"] = []
        all_metric_lists["map"] = []
        
        for item in self.test_data:
            query = item["query"]
            relevant_ids = set(item["relevant_ids"])
            relevance_scores = item.get("relevance_scores", {})
            
            # 使用混合检索器，一次返回最大 k 个结果
            docs = hybrid_retriever.retrieve(query, top_k=max(k_values))
            retrieved_ids = _extract_doc_ids(docs)
            
            metrics = calculate_all_metrics(retrieved_ids, relevant_ids, relevance_scores, k_values)
            
            for k in k_values:
                all_metric_lists[f"recall@{k}"].append(metrics[f"recall@{k}"])
            all_metric_lists["mrr"].append(metrics["mrr"])
            all_metric_lists["ndcg@10"].append(metrics["ndcg@10"])
            all_metric_lists["map"].append(metrics["map"])
        
        avg_metrics = {k: sum(v)/len(v) if v else 0.0 for k, v in all_metric_lists.items()}
        logger.info(f"Optimized 结果: {avg_metrics}")
        return avg_metrics