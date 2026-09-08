"""稀疏向量/BM25 索引服务 - 用于混合检索中的关键词匹配"""

from typing import List, Tuple, Optional
from loguru import logger
from langchain_core.documents import Document

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("请安装 rank-bm25: pip install rank-bm25")

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("未安装 jieba，将使用空格分词（中文效果较差）")


class SparseEmbeddingService:
    """
    BM25 稀疏向量索引服务（单例）
    负责从 Milvus 加载文档，构建 BM25 索引，并提供关键词检索
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

        self.bm25_index: Optional[BM25Okapi] = None
        self.doc_id_to_doc: dict[str, Document] = {}
        self.doc_ids: list[str] = []
        self._initialized = True
        self._is_built = False

    def _tokenize(self, text: str) -> List[str]:
        """分词，支持 jieba 中文分词"""
        if JIEBA_AVAILABLE:
            return list(jieba.cut(text))
        else:
            # 简单按空格和标点切分
            for char in "，。、；：！？（）《》\"'":
                text = text.replace(char, " ")
            return text.split()

    def build_index(self, force_rebuild: bool = False) -> None:
        """从 Milvus 加载文档并构建 BM25 索引"""
        if self._is_built and not force_rebuild:
            logger.debug("BM25 索引已存在，跳过构建")
            return

        from app.services.vector_store_manager import vector_store_manager

        logger.info("开始构建 BM25 索引...")

        if not vector_store_manager.is_available():
            raise RuntimeError("向量存储不可用，无法构建 BM25 索引")

        # 获取 MilvusClient 实例
        client = vector_store_manager.vector_store.client
        collection_name = vector_store_manager.collection_name

        # 查询所有文档（兼容不同版本的参数名）
        try:
            results = client.query(
                collection_name=collection_name,
                filter="",
                output_fields=["id", "content", "metadata"],
                limit=10000
            )
        except TypeError:
            results = client.query(
                collection_name=collection_name,
                expr="",
                output_fields=["id", "content", "metadata"],
                limit=10000
            )

        if not results:
            logger.warning("Milvus 中没有文档，BM25 索引为空")
            self._is_built = True
            return

        # 构建语料库
        corpus = []
        self.doc_ids = []
        self.doc_id_to_doc = {}

        for item in results:
            doc_id = item["id"]
            content = item["content"]
            metadata = item.get("metadata", {})
            doc = Document(page_content=content, metadata=metadata)
            self.doc_id_to_doc[doc_id] = doc
            self.doc_ids.append(doc_id)
            corpus.append(content)

        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self._is_built = True
        logger.info(f"BM25 索引构建完成，共 {len(self.doc_ids)} 个文档")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """BM25 检索，返回 (doc_id, score) 列表"""
        if not self._is_built:
            self.build_index()

        if self.bm25_index is None or not self.doc_ids:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [(self.doc_ids[i], scores[i]) for i in top_indices if scores[i] > 0]

    def refresh(self) -> None:
        """强制重建索引（文档增删后调用）"""
        self.build_index(force_rebuild=True)

    def get_document(self, doc_id: str) -> Optional[Document]:
        """根据 ID 获取文档"""
        return self.doc_id_to_doc.get(doc_id)

    def get_documents(self, doc_ids: List[str]) -> List[Document]:
        """批量获取文档"""
        return [self.doc_id_to_doc[doc_id] for doc_id in doc_ids if doc_id in self.doc_id_to_doc]


# 全局单例
sparse_embedding_service = SparseEmbeddingService()