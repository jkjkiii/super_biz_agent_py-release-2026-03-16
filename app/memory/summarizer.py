"""摘要压缩服务

将旧对话压缩成精炼摘要，保留核心事实和主线
"""

from typing import List, Dict, Any
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen

from app.config import config


class Summarizer:
    """对话摘要压缩器"""

    def __init__(self):
        self.llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0.3,  # 低温度，保证摘要稳定
        )
        self.prompt = ChatPromptTemplate.from_template("""
你是一个专业的对话摘要专家。请将以下对话压缩成一段精炼的摘要（80-150字）。

要求：
1. 保留**关键事实**：用户提到的配置、错误码、服务名、版本号等
2. 保留**问题经过**：发生了什么、做了什么尝试、结果如何
3. 保留**用户偏好**：用户表达的习惯、偏好
4. 忽略日常问候和无关闲聊
5. 摘要要连贯，像一段叙述，而不是列表

对话内容：
{conversation}

摘要：
""")

    async def summarize(self, messages: List[Dict[str, Any]]) -> str:
        """
        压缩对话为摘要

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}, ...]

        Returns:
            str: 摘要文本
        """
        if not messages:
            return ""

        # 把消息列表转成文本
        conv_text = "\n".join([
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in messages
            if m.get('role') != 'system'  # 跳过系统消息
        ])

        if not conv_text.strip():
            return ""

        logger.debug(f"开始摘要压缩，原始消息数: {len(messages)}，文本长度: {len(conv_text)}")

        try:
            chain = self.prompt | self.llm
            response = await chain.ainvoke({"conversation": conv_text})
            summary = response.content.strip()

            # 如果摘要太长或太短，做简单处理
            if len(summary) > 300:
                summary = summary[:300] + "..."
            if len(summary) < 10 and len(messages) > 2:
                # 摘要太短，可能是 LLM 没理解，用简单方法
                logger.warning(f"摘要过短（{len(summary)}字符），使用简单摘要")
                summary = f"用户讨论了关于 {', '.join([m.get('content', '')[:20] for m in messages if m.get('role')=='user'][:3])} 的问题"

            logger.debug(f"摘要压缩完成，长度: {len(summary)} 字符")
            return summary

        except Exception as e:
            logger.error(f"摘要压缩失败: {e}")
            # 降级：返回简单摘要
            user_messages = [m.get('content', '') for m in messages if m.get('role') == 'user']
            if user_messages:
                return f"用户讨论了以下问题: {'; '.join([u[:30] for u in user_messages[:2]])}"
            return ""

    async def summarize_with_metadata(
        self,
        messages: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        摘要压缩并附带元数据

        Returns:
            {"summary": "...", "fact_count": 3, "key_points": [...]}
        """
        summary = await self.summarize(messages)
        return {
            "summary": summary,
            "fact_count": len([m for m in messages if m.get('role') == 'user']),
            "key_points": summary.split("。")[:3] if summary else [],
        }