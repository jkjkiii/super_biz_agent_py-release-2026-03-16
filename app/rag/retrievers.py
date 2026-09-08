"""RAG 检索器 - 混合检索、多查询检索等策略"""

from typing import List, Optional, Tuple
from collections import defaultdict
from loguru import logger
from langchain_core.documents import Document

from app.config import config
from app.services.vector_store_manager import vector_store_manager
from app.services.sparse_embedding_service import sparse_embedding_service
from app.services.reranker_service import reranker_service


class HybridRetriever:
    """
    混合检索器（BM25 + 向量检索 + RRF 融合 + 可选重排序）
    """

    def __init__(
        self,
        bm25_top_k: Optional[int] = None,
        vector_top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        candidate_top_k: Optional[int] = None,
        enable_rerank: Optional[bool] = None,
        final_top_k: Optional[int] = None,
    ):
        # 从配置读取默认值，若未传入则使用配置
        self.bm25_top_k = bm25_top_k if bm25_top_k is not None else config.bm25_top_k
        self.vector_top_k = vector_top_k if vector_top_k is not None else config.vector_top_k
        self.rrf_k = rrf_k if rrf_k is not None else config.rrf_k
        self.candidate_top_k = candidate_top_k if candidate_top_k is not None else config.candidate_top_k
        self.enable_rerank = enable_rerank if enable_rerank is not None else config.reranker_enabled
        self.final_top_k = final_top_k if final_top_k is not None else config.final_top_k

        if not vector_store_manager.is_available():
            logger.warning("向量存储不可用，混合检索器将无法工作")

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[str, float]],
        vector_results: List[Tuple[str, float]],
        k: int = 60
    ) -> List[str]:
        """RRF 融合两个排名列表"""
        scores = defaultdict(float)

        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            scores[doc_id] += 1.0 / (k + rank)

        for rank, (doc_id, _) in enumerate(vector_results, start=1):
            scores[doc_id] += 1.0 / (k + rank)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        enable_rerank: Optional[bool] = None,
    ) -> List[Document]:
        """执行混合检索"""
        final_top_k = top_k or self.final_top_k
        use_rerank = enable_rerank if enable_rerank is not None else self.enable_rerank

        # 1. 构建 BM25 索引
        sparse_embedding_service.build_index()

        # 2. BM25 检索
        bm25_results = sparse_embedding_service.search(query, top_k=self.bm25_top_k)
        logger.debug(f"BM25 检索: {len(bm25_results)} 个结果")

        # 3. 向量检索
        if not vector_store_manager.is_available():
            logger.warning("向量存储不可用，仅使用 BM25")
            vector_results = []
        else:
            docs_with_scores = vector_store_manager.vector_store.similarity_search_with_score(
                query, k=self.vector_top_k
            )
            vector_results = []
            for doc, score in docs_with_scores:
                doc_id = doc.metadata.get("id")
                if doc_id:
                    vector_results.append((doc_id, score))
            logger.debug(f"向量检索: {len(vector_results)} 个结果")

        # 4. RRF 融合
        fused_ids = self._reciprocal_rank_fusion(
            bm25_results, vector_results, k=self.rrf_k
        )

        if not fused_ids:
            logger.warning("未检索到任何文档")
            return []

        # 5. 获取候选文档
        candidate_ids = fused_ids[:self.candidate_top_k]
        candidate_docs = []
        for doc_id in candidate_ids:
            doc = sparse_embedding_service.get_document(doc_id)
            if doc:
                candidate_docs.append(doc)

        if not candidate_docs:
            return []

        # 6. 重排序（可选）
        if use_rerank:
            reranked_docs = reranker_service.rerank(query, candidate_docs, top_k=final_top_k)
            return reranked_docs
        else:
            return candidate_docs[:final_top_k]

    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """检索并返回相关性分数（仅在启用重排序时有效）"""
        final_top_k = top_k or self.final_top_k
        docs = self.retrieve(query, top_k=self.candidate_top_k, enable_rerank=False)
        if not docs:
            return []
        return reranker_service.rerank_with_scores(query, docs, top_k=final_top_k)


class MultiQueryRetriever:
    """多查询检索器（占位实现）"""
    def __init__(self, base_retriever: HybridRetriever, num_queries: int = 3, final_top_k: int = 5):
        self.base_retriever = base_retriever
        self.num_queries = num_queries
        self.final_top_k = final_top_k

    def retrieve(self, query: str) -> List[Document]:
        # TODO: 调用 LLM 生成查询变体
        logger.warning("MultiQueryRetriever 尚未实现 LLM 查询改写，使用单查询")
        return self.base_retriever.retrieve(query, top_k=self.final_top_k)


# 创建全局实例（自动从 config 读取参数）
hybrid_retriever = HybridRetriever()
multi_query_retriever = MultiQueryRetriever(hybrid_retriever)