"""向量存储管理器 - 封装 Milvus VectorStore 操作"""

from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger

from app.config import config
from app.services.vector_embedding_service import vector_embedding_service


# 统一使用 biz collection
COLLECTION_NAME = "biz"# 操作的表名


class VectorStoreManager:
    """向量存储管理器（延迟初始化）"""

    def __init__(self):
        """初始化向量存储管理器（不立即连接数据库）"""
        self.vector_store = None
        self.collection_name = COLLECTION_NAME
        self._initialized = False
        self._init_error = None

    def _initialize_vector_store(self):
        """初始化 Milvus VectorStore（首次调用时懒加载）"""
        if self._initialized:
            return True

        try:
            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
            }

            # 创建 LangChain Milvus VectorStore
            self.vector_store = Milvus(
                embedding_function=vector_embedding_service,
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=False,
                drop_old=False,
                text_field="content",# 定义各列的名字
                vector_field="vector",
                primary_field="id",
                metadata_field="metadata",
            )

            self._initialized = True
            self._init_error = None
            logger.info(
                f"VectorStore 初始化成功: {config.milvus_host}:{config.milvus_port}, "
                f"collection: {self.collection_name}"
            )
            return True

        except Exception as e:
            self._init_error = str(e)
            logger.error(f"VectorStore 初始化失败: {e}")
            return False

    def is_available(self) -> bool:
        """检查向量存储是否可用"""
        return self._initialize_vector_store()

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        批量添加文档到向量存储（自动批量向量化）

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档 ID 列表
        """
        if not self.is_available():
            raise RuntimeError(f"向量存储不可用: {self._init_error}")
        try:
            import time
            import uuid
            start_time = time.time()
            
            # 为每个文档生成唯一 id（因为 auto_id=False）
            ids = [str(uuid.uuid4()) for _ in documents]
            
            # ========== 关键修改：将 id 存入 metadata ==========
            for doc, id_ in zip(documents, ids):
                # 确保 metadata 是 dict，并添加 'id' 字段
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["id"] = id_   # 添加这一行
            # =================================================
            
            
            # LangChain Milvus 的 add_documents 会自动调用 embedding_function
            # 并进行批量处理，性能更好
            result_ids = self.vector_store.add_documents(documents, ids=ids)
            
            elapsed = time.time() - start_time
            logger.info(
                f"批量添加 {len(documents)} 个文档到 VectorStore 完成, "
                f"耗时: {elapsed:.2f}秒, 平均: {elapsed/len(documents):.2f}秒/个"
            )
            return result_ids
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    def delete_by_source(self, file_path: str) -> int:
        """
        删除指定文件的所有文档

        Args:
            file_path: 文件路径

        Returns:
            int: 删除的文档数量
        """
        if not self.is_available():
            logger.warning(f"向量存储不可用，跳过删除: {file_path}")
            return 0
        try:
            # 使用 milvus_manager 获取已连接的 collection
            from app.core.milvus_client import milvus_manager
            collection = milvus_manager.get_collection()
            
            # metadata 是 JSON 字段，使用 JSON 路径查询语法
            # _source 是文档的来源文件路径
            expr = f'metadata["_source"] == "{file_path}"'
            
            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0
            
            logger.info(f"删除文件旧数据: {file_path}, 删除数量: {deleted_count}")
            return deleted_count
            
        except Exception as e:
            logger.warning(f"删除旧数据失败 (可能是首次索引): {e}")
            return 0

    def get_vector_store(self) -> Milvus:
        """
        获取 VectorStore 实例

        Returns:
            Milvus: VectorStore 实例

        Raises:
            RuntimeError: 向量存储不可用时抛出
        """
        if not self.is_available():
            raise RuntimeError(f"向量存储不可用: {self._init_error}")
        return self.vector_store

    def similarity_search(self, query: str, k: int = 50) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表
        """
        if not self.is_available():
            logger.warning("向量存储不可用，搜索返回空结果")
            return []
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"相似度搜索完成: query='{query}', 结果数={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []


# 全局单例
vector_store_manager = VectorStoreManager()
