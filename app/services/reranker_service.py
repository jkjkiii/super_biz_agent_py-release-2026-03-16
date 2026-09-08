"""重排序服务 - 使用 Cross-Encoder 模型对候选文档精排"""

import os
from typing import List, Optional, Tuple
from loguru import logger
from langchain_core.documents import Document

from app.config import config

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")


class RerankerService:
    """
    重排序服务（单例）
    使用 Cross-Encoder 模型对文档进行精细排序
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 从配置读取模型名称
        self.model_name: str = config.reranker_model
        self.model: Optional[CrossEncoder] = None
        self._initialized = True
        self._is_loaded = False

    def load_model(self, model_name: Optional[str] = None) -> None:
        """懒加载模型"""
        if self._is_loaded:
            return

        # 设置 Hugging Face 镜像（国内加速）
        if config.hf_endpoint:
            os.environ["HF_ENDPOINT"] = config.hf_endpoint
            logger.info(f"使用 HF 镜像: {config.hf_endpoint}")

        if model_name:
            self.model_name = model_name
        else:
            self.model_name = config.reranker_model

        logger.info(f"正在加载重排序模型: {self.model_name}")
        try:
            self.model = CrossEncoder(self.model_name, max_length=512)
            self._is_loaded = True
            logger.info("重排序模型加载完成")
        except Exception as e:
            logger.error(f"重排序模型加载失败: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """重排序返回 top_k 个文档"""
        if not documents:
            return []

        if top_k >= len(documents):
            return documents

        self.load_model()

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        sorted_pairs = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True
        )
        reranked_docs = [doc for _, doc in sorted_pairs[:top_k]]

        logger.debug(
            f"重排序完成: 输入 {len(documents)} 个文档, 返回 {len(reranked_docs)} 个"
        )
        return reranked_docs

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """重排序并返回分数"""
        if not documents:
            return []

        if top_k >= len(documents):
            top_k = len(documents)

        self.load_model()

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        sorted_pairs = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True
        )

        return [(doc, score) for score, doc in sorted_pairs[:top_k]]


# 全局单例
reranker_service = RerankerService()