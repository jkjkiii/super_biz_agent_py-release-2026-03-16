"""长期记忆 - BGE 向量存储

使用 Milvus 存储用户历史事实，按 user_id 隔离
"""

import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
from langchain_core.documents import Document

from app.config import config

# ==================== 时间衰减配置 ====================
# 衰减系数：值越大，旧数据衰减越快
# 0.01 表示每天衰减约 1%（一个月后权重约 74%）
# 0.02 表示每天衰减约 2%（一个月后权重约 55%）
DECAY_RATE = 0.02  # 可调，建议 0.005~0.02

# 最低有效权重：低于此值的记忆不再召回
# 0.1 表示重要性打分低于 0.1 的不召回
MIN_EFFECTIVE_IMPORTANCE = 0.1
# =================================================


class LongTermMemory:
    """
    长期记忆服务（BGE 向量检索）

    特点：
    1. 复用现有的 vector_store_manager
    2. 使用独立的 collection: "memory"
    3. 按 user_id 隔离（跨会话共享记忆）
    4. 存储格式：{"content": "事实", "metadata": {"user_id": "...", "importance": 0.9, "timestamp": "..."}}
    5. 检索时动态计算时间衰减，永不物理删除数据
    """

    MEMORY_COLLECTION = "memory"  # 专用 collection
    VECTOR_DIM = 1024
    ID_MAX_LENGTH = 100
    CONTENT_MAX_LENGTH = 8000

    def __init__(self):
        self.collection_name = self.MEMORY_COLLECTION
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """确保 memory collection 存在（复用 biz 的创建逻辑）"""
        from pymilvus import utility, Collection, CollectionSchema, FieldSchema, DataType
        from app.core.milvus_client import milvus_manager

        try:
            # 确保 Milvus 已连接
            if milvus_manager._client is None:
                milvus_manager.connect()

            # 检查 collection 是否存在
            if utility.has_collection(self.MEMORY_COLLECTION):
                logger.info(f"✅ 长期记忆 collection 已存在: {self.MEMORY_COLLECTION}")
                return

            logger.info(f"正在创建长期记忆 collection: {self.MEMORY_COLLECTION}")

            # 定义字段（与 biz 完全一致）
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    max_length=self.ID_MAX_LENGTH,
                    is_primary=True,
                ),
                FieldSchema(
                    name="vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.VECTOR_DIM,
                ),
                FieldSchema(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=self.CONTENT_MAX_LENGTH,
                ),
                FieldSchema(
                    name="metadata",
                    dtype=DataType.JSON,
                ),
            ]

            # 创建 schema
            schema = CollectionSchema(
                fields=fields,
                description="Long-term memory collection",
                enable_dynamic_field=False,
            )

            # 创建 collection
            collection = Collection(
                name=self.MEMORY_COLLECTION,
                schema=schema,
                num_shards=2,
            )

            # 创建索引（复用 biz 的索引配置）
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            collection.create_index(
                field_name="vector",
                index_params=index_params,
            )

            # 加载 collection
            collection.load()

            logger.info(f"✅ 长期记忆 collection 创建成功: {self.MEMORY_COLLECTION}")

        except Exception as e:
            logger.error(f"❌ 创建 memory collection 失败: {e}")
            raise

    async def add_fact(
        self,
        content: str,
        user_id: str,
        importance: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """存入一条事实到长期记忆"""
        from datetime import datetime
        import uuid
        from app.core.milvus_client import milvus_manager
        from app.services.vector_embedding_service import vector_embedding_service

        doc_metadata = {
            "user_id": user_id,
            "importance": importance,
            "timestamp": datetime.now().isoformat(),
            "type": "memory",
        }
        if metadata:
            doc_metadata.update(metadata)

        doc_id = f"mem_{uuid.uuid4().hex[:16]}"

        # 获取 embedding
        embedding = await vector_embedding_service.aembed_query(content)

        client = milvus_manager._client
        if client is None:
            milvus_manager.connect()
            client = milvus_manager._client

        data = [{
            "id": doc_id,
            "vector": embedding,
            "content": content,
            "metadata": doc_metadata,
        }]

        try:
            client.insert(collection_name=self.MEMORY_COLLECTION, data=data)
            logger.info(f"[记忆] 已存入: user_id={user_id}, importance={importance:.2f}, content={content[:50]}...")
            return doc_id
        except Exception as e:
            logger.error(f"[记忆] 存入失败: {e}")
            return ""

    # ===================== 核心：时间衰减检索 =====================

    async def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 3,
        min_importance: float = 0.5,
        enable_time_decay: bool = True,
    ) -> List[Document]:
        """
        检索长期记忆，支持时间衰减（方案一：检索时动态计算）

        核心流程：
        1. 按 user_id 和 min_importance 过滤，多召回一些候选（top_k * 3）
        2. 对每个候选计算时间衰减后的有效权重
        3. 按有效权重重新排序，取前 top_k 个

        设计理念：
        - 用户数据是宝贵资产，永不物理删除
        - 旧数据权重自然衰减，但不丢失
        - 可随时调整衰减系数，重新评估历史数据
        """
        try:
            from app.core.milvus_client import milvus_manager
            from app.services.vector_embedding_service import vector_embedding_service

            query_embedding = await vector_embedding_service.aembed_query(query)

            client = milvus_manager._client
            if client is None:
                milvus_manager.connect()
                client = milvus_manager._client

            # 基础过滤条件
            expr = f'metadata["user_id"] == "{user_id}" AND metadata["importance"] >= {min_importance}'

            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

            # ========== 多召回一些候选（给衰减留出空间） ==========
            # 如果启用衰减，多召回 3 倍；否则正常召回 top_k
            limit = top_k * 3 if enable_time_decay else top_k

            results = client.search(
                collection_name=self.MEMORY_COLLECTION,
                data=[query_embedding],
                anns_field="vector",
                search_params=search_params,
                limit=limit,
                filter=expr,
                output_fields=["content", "metadata"],
            )

            docs = []
            if results and len(results) > 0:
                now = datetime.now()

                for hit in results[0]:
                    content = hit.get("entity", {}).get("content", "")
                    metadata = hit.get("entity", {}).get("metadata", {})

                    # ========== 解析时间戳和重要性 ==========
                    timestamp_str = metadata.get("timestamp", "")
                    importance = metadata.get("importance", 0.5)
                    effective_importance = importance

                    if enable_time_decay and timestamp_str:
                        try:
                            # 计算年龄（天）
                            stored_time = datetime.fromisoformat(timestamp_str)
                            age_days = (now - stored_time).total_seconds() / (24 * 3600)

                            # ===== 指数衰减公式：有效权重 = 原始重要性 × e^(-衰减系数 × 天数) =====
                            decay_factor = math.exp(-DECAY_RATE * age_days)
                            effective_importance = importance * decay_factor

                            # 记录衰减信息（用于调试和展示）
                            metadata["original_importance"] = round(importance, 3)
                            metadata["decay_factor"] = round(decay_factor, 3)
                            metadata["effective_importance"] = round(effective_importance, 3)
                            metadata["age_days"] = round(age_days, 1)

                        except Exception as e:
                            logger.warning(f"时间解析失败: {e}，使用原始重要性")
                            effective_importance = importance

                    # ========== 低于最低有效权重的，直接跳过 ==========
                    if effective_importance < MIN_EFFECTIVE_IMPORTANCE:
                        logger.debug(
                            f"[记忆] 跳过低有效权重: "
                            f"content={content[:30]}..., "
                            f"原始={importance:.2f}, "
                            f"有效={effective_importance:.2f}"
                        )
                        continue

                    # 更新 metadata 中的 importance 为有效权重（用于后续排序）
                    metadata["importance"] = effective_importance

                    doc = Document(page_content=content, metadata=metadata)
                    docs.append(doc)

                # ========== 按有效权重（即 importance）降序排序 ==========
                docs.sort(
                    key=lambda d: d.metadata.get("importance", 0),
                    reverse=True
                )

                # 只取前 top_k 个
                docs = docs[:top_k]

            logger.info(
                f"[记忆] 检索完成: "
                f"query={query[:30]}, "
                f"user_id={user_id}, "
                f"结果数={len(docs)}, "
                f"衰减={'启用' if enable_time_decay else '禁用'}"
            )
            return docs

        except Exception as e:
            logger.warning(f"[记忆] 检索失败: {e}")
            return []

    async def retrieve_as_context(
        self,
        query: str,
        user_id: str,
        top_k: int = 3,
        enable_time_decay: bool = True,
    ) -> str:
        """
        检索记忆并格式化为上下文文本

        Args:
            query: 查询文本
            user_id: 用户ID
            top_k: 返回数量
            enable_time_decay: 是否启用时间衰减

        Returns:
            str: 格式化的上下文文本
        """
        docs = await self.retrieve(query, user_id, top_k, enable_time_decay=enable_time_decay)

        if not docs:
            return ""

        context_parts = ["【历史记忆】"]
        for i, doc in enumerate(docs, 1):
            content = doc.page_content
            source = doc.metadata.get("timestamp", "未知时间")
            importance = doc.metadata.get("importance", 0.5)
            age_days = doc.metadata.get("age_days", "N/A")

            # 显示有效权重和年龄（便于调试，也告诉用户信息的新旧程度）
            context_parts.append(
                f"{i}. {content}（时间: {source}，相关度: {importance:.2f}，距今: {age_days}天）"
            )

        return "\n".join(context_parts)

    async def clear_user_memories(self, user_id: str) -> int:
        """清空某个用户的所有记忆（慎用）"""
        try:
            from app.core.milvus_client import milvus_manager

            client = milvus_manager._client
            if client is None:
                milvus_manager.connect()
                client = milvus_manager._client

            expr = f'metadata["user_id"] == "{user_id}"'
            result = client.delete(collection_name=self.MEMORY_COLLECTION, filter=expr)
            count = result.delete_count if hasattr(result, "delete_count") else 0
            logger.warning(f"[记忆] 清空用户: {user_id}, 删除 {count} 条")
            return count
        except Exception as e:
            logger.error(f"[记忆] 清空失败: {e}")
            return 0