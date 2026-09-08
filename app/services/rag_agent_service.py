"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""

from typing import Annotated, Any, AsyncGenerator, Dict, Sequence, Optional

from langchain.agents import create_agent
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from loguru import logger
from typing_extensions import TypedDict
from langchain_qwq import ChatQwen

from app.config import config
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry
from app.core.intent_router import intent_router, Intent
from app.core.route_handlers import route_handlers

# ========== 默认用户 ID（前端不传时使用） ==========
DEFAULT_USER_ID = "default_user"
# =================================================


class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def trim_messages_middleware(state: AgentState) -> dict[str, Any] | None:
    """
    修剪消息历史，只保留最近的几条消息以适应上下文窗口

    策略：
    - 保留第一条系统消息（System Message）
    - 保留最近的 6 条消息（3 轮对话）
    - 当消息少于等于 7 条时，不做修剪

    Args:
        state: Agent 状态

    Returns:
        包含修剪后消息的字典，如果无需修剪则返回 None
    """
    messages = state["messages"]

    # 如果消息数量较少，无需修剪
    if len(messages) <= 7:
        return None

    # 提取第一条系统消息
    first_msg = messages[0]

    # 保留最近的 6 条消息（确保包含完整的对话轮次）
    recent_messages = messages[-6:] if len(messages) % 2 == 0 else messages[-7:]

    # 构建新的消息列表
    new_messages = [first_msg] + list(recent_messages)

    logger.debug(f"修剪消息历史: {len(messages)} -> {len(new_messages)} 条")

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatQwen 原生集成"""

    def __init__(self, streaming: bool = True):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()

        self.model = ChatQwen(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            temperature=0.7,
            streaming=streaming,
        )

        # 定义基础工具
        self.tools = [retrieve_knowledge, get_current_time]

        # MCP 客户端（延迟初始化，使用全局管理）
        self.mcp_tools: list = []

        # 创建内存检查点（用于会话管理）
        self.checkpointer = MemorySaver()

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False

        logger.info(f"RAG Agent 服务初始化完成 (ChatQwen), model={self.model_name}, streaming={streaming}")

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具）"""
        if self._agent_initialized:
            return

        # 使用全局 MCP 客户端管理器（带重试拦截器）
        mcp_client = await get_mcp_client_with_retry()

        # 获取 MCP 工具
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")

        # 将 MCP 工具添加到实例变量中
        self.mcp_tools = mcp_tools

        # 合并所有工具
        all_tools = self.tools + self.mcp_tools

        self.agent = create_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
        )

        self._agent_initialized = True

        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Returns:
            str: 系统提示词
        """
        from textwrap import dedent

        return dedent("""
            你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。

            工作原则:
            1. 理解用户需求，选择合适的工具来完成任务
            2. 当需要获取实时信息或专业知识时，主动使用相关工具
            3. 基于工具返回的结果提供准确、专业的回答
            4. 如果工具无法提供足够信息，请诚实地告知用户

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()

    # ===================== 核心改动：统一记录短期记忆 =====================

    async def query(self, question: str, session_id: str, user_id: Optional[str] = None) -> str:
        """非流式 - 支持硬路由"""
        try:
            await self._initialize_agent()

            # 1. 意图识别
            route_result = intent_router.route(question)
            logger.info(f"[会话 {session_id}] 意图: {route_result.intent.value}, 置信度: {route_result.confidence:.2f}")

            actual_user_id = user_id or DEFAULT_USER_ID
            logger.info(f"[会话 {session_id}] 用户标识: {actual_user_id} (来源: {'前端' if user_id else '默认'})")

            # ========== 🆕 立即记录用户问题到短期记忆（无论置信度） ==========
            if actual_user_id:
                route_handlers._record_user_question(actual_user_id, question)

            # ====== 硬路由：所有场景都传入 actual_user_id ======

            # 闲聊 → 直出
            if route_result.intent == Intent.CHAT and route_result.confidence > 0.6:
                answer = await route_handlers.handle_chat(question, actual_user_id)
                if actual_user_id:
                    route_handlers._record_assistant_answer(actual_user_id, answer)
                return answer

            # 知识查询 → 快速 RAG
            if route_result.intent == Intent.KNOWLEDGE and route_result.confidence > 0.7:
                answer = await route_handlers.handle_knowledge(question, actual_user_id)
                if actual_user_id:
                    route_handlers._record_assistant_answer(actual_user_id, answer)
                return answer

            # 系统控制 → 直接执行
            if route_result.intent == Intent.CONTROL and route_result.confidence > 0.7:
                answer = await route_handlers.handle_control(question, actual_user_id)
                if actual_user_id:
                    route_handlers._record_assistant_answer(actual_user_id, answer)
                return answer

            # 故障排查 → 走硬路由
            if route_result.intent == Intent.TROUBLESHOOT:
                logger.info(f"[会话 {session_id}] 🟢 硬路由 → 故障排查")
                answer = await route_handlers.handle_troubleshoot(question, actual_user_id)
                if actual_user_id:
                    route_handlers._record_assistant_answer(actual_user_id, answer)
                return answer

            # ========== 兜底：走 Agent ==========
            logger.info(f"[会话 {session_id}] 🔵 兜底 → 标准 Agent")
            messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=question)]
            agent_input = {"messages": messages}
            config_dict = {"configurable": {"thread_id": session_id}}
            result = await self.agent.ainvoke(input=agent_input, config=config_dict)
            answer = result["messages"][-1].content

            if actual_user_id:
                route_handlers._record_assistant_answer(actual_user_id, answer)

            return answer

        except Exception as e:
            logger.error(f"查询失败: {e}")
            return f"处理失败: {str(e)}"

    # ===================== 流式接口：同样统一记录短期记忆 =====================

    async def query_stream(
        self,
        question: str,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式 - 支持硬路由"""
        try:
            await self._initialize_agent()

            # 意图识别
            route_result = intent_router.route(question)
            logger.info(f"[会话 {session_id}] 意图识别: {route_result.intent.value}, 置信度: {route_result.confidence:.2f}")

            actual_user_id = user_id or DEFAULT_USER_ID

            # ========== 流式：先记录用户问题 ==========
            if actual_user_id:
                route_handlers._record_user_question(actual_user_id, question)

            # 先 yield 意图信息
            yield {
                "type": "intent",
                "data": {
                    "intent": route_result.intent.value,
                    "confidence": route_result.confidence
                }
            }

            full_answer = ""

            # ====== 故障排查（流式） ======
            if route_result.intent == Intent.TROUBLESHOOT:
                logger.info(f"[会话 {session_id}] 🟢 硬路由 → 故障排查（流式）")
                result = await route_handlers.handle_troubleshoot(question, actual_user_id)
                full_answer = result
                yield {"type": "content", "data": result}
                yield {"type": "complete"}
                if actual_user_id:
                    route_handlers._record_assistant_answer(actual_user_id, full_answer)
                return

            # ====== 闲聊直出（流式） ======
            if route_result.intent == Intent.CHAT and route_result.confidence > 0.6:
                logger.info(f"[会话 {session_id}] 🟢 硬路由 → 闲聊直出（流式）")
                result = await route_handlers.handle_chat(question, actual_user_id)
                full_answer = result
                yield {"type": "content", "data": result}
                yield {"type": "complete"}
                if actual_user_id:
                    route_handlers._record_assistant_answer(actual_user_id, full_answer)
                return

            # ====== 知识快速 RAG（流式） ======
            if route_result.intent == Intent.KNOWLEDGE and route_result.confidence > 0.7:
                logger.info(f"[会话 {session_id}] 🟢 硬路由 → 知识快速 RAG（流式）")
                result = await route_handlers.handle_knowledge(question, actual_user_id)
                full_answer = result
                yield {"type": "content", "data": result}
                yield {"type": "complete"}
                if actual_user_id:
                    route_handlers._record_assistant_answer(actual_user_id, full_answer)
                return

            # ====== 系统控制（流式） ======
            if route_result.intent == Intent.CONTROL and route_result.confidence > 0.7:
                logger.info(f"[会话 {session_id}] 🟢 硬路由 → 系统控制（流式）")
                result = await route_handlers.handle_control(question, actual_user_id)
                full_answer = result
                yield {"type": "content", "data": result}
                yield {"type": "complete"}
                if actual_user_id:
                    route_handlers._record_assistant_answer(actual_user_id, full_answer)
                return

            # ====== 兜底：走 Agent 流式 ======
            logger.info(f"[会话 {session_id}] 🔵 兜底 → 标准 Agent（流式）")
            enhanced_system_prompt = f"{self.system_prompt}\n\n【意图信息】识别为用户意图: {route_result.intent.value}，置信度: {route_result.confidence:.2f}"

            messages = [
                SystemMessage(content=enhanced_system_prompt),
                HumanMessage(content=question)
            ]

            agent_input = {"messages": messages}
            config_dict = {"configurable": {"thread_id": session_id}}

            async for token, metadata in self.agent.astream(
                input=agent_input,
                config=config_dict,
                stream_mode="messages",
            ):
                node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                message_type = type(token).__name__

                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, 'content_blocks', None)
                    if content_blocks and isinstance(content_blocks, list):
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_content = block.get('text', '')
                                if text_content:
                                    full_answer += text_content
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name
                                    }

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            yield {"type": "complete"}
            if actual_user_id and full_answer:
                route_handlers._record_assistant_answer(actual_user_id, full_answer)

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（流式）: {e}")
            yield {"type": "error", "data": str(e)}
            raise

    def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史（从 MemorySaver checkpointer 中读取）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        try:
            # 使用 checkpointer 的 get 方法获取最新的检查点
            config = {"configurable": {"thread_id": session_id}}

            # 获取该 thread 的最新检查点
            checkpoint_tuple = self.checkpointer.get(config)

            if not checkpoint_tuple:
                logger.info(f"获取会话历史: {session_id}, 消息数量: 0")
                return []

            # checkpoint_tuple 可能是命名元组或普通元组，安全地提取 checkpoint
            # 通常第一个元素是 checkpoint 数据
            if hasattr(checkpoint_tuple, 'checkpoint'):
                checkpoint_data = checkpoint_tuple.checkpoint  # type: ignore
            else:
                # 如果是普通元组，第一个元素是 checkpoint
                checkpoint_data = checkpoint_tuple[0] if checkpoint_tuple else {}

            # 从检查点中提取消息
            messages = checkpoint_data.get("channel_values", {}).get("messages", [])

            # 转换为前端需要的格式
            history = []
            for msg in messages:
                # 跳过系统消息
                if isinstance(msg, SystemMessage):
                    continue

                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)

                # 提取时间戳（如果有的话）
                timestamp = getattr(msg, 'timestamp', None)
                if timestamp:
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    })
                else:
                    from datetime import datetime
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    })

            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
            return history

        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    def clear_session(self, session_id: str) -> bool:
        """
        清空会话历史（从 MemorySaver checkpointer 中删除）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            bool: 是否成功
        """
        try:
            # 使用 checkpointer 的 delete_thread 方法删除该 thread 的所有检查点
            self.checkpointer.delete_thread(session_id)

            logger.info(f"已清除会话历史: {session_id}")
            return True

        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出
rag_agent_service = RagAgentService(streaming=True)