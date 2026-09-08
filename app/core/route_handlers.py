"""硬路由处理器 - 不同意图走不同的处理管道"""

from typing import List, Optional, Dict, Any
from loguru import logger
from datetime import datetime

from app.config import config
from app.rag.retrievers import hybrid_retriever

from app.memory.short_term import ShortTermMemory
from app.memory.summarizer import Summarizer
from app.memory.importance_scorer import ImportanceScorer
from app.memory.long_term import LongTermMemory


class RouteHandlers:
    """各意图的硬路由处理器"""

    def __init__(self):
        
        print(f"[DEBUG] config.enable_memory = {config.enable_memory}")
        logger.info(f"[DEBUG] config.enable_memory = {config.enable_memory}")
        
        from langchain_community.chat_models import ChatTongyi
        from langchain_core.prompts import ChatPromptTemplate

        self.llm = ChatTongyi(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0.7,
        )
        self.prompt_template = ChatPromptTemplate

        # MCP 客户端（延迟初始化）
        self._mcp_client = None

        # ========== 初始化记忆系统 ==========
        self.enable_memory = config.enable_memory if hasattr(config, 'enable_memory') else False

        if self.enable_memory:
            self.long_term = LongTermMemory()
            self.summarizer = Summarizer()
            self.importance_scorer = ImportanceScorer(
                threshold=config.memory_importance_threshold if hasattr(config, 'memory_importance_threshold') else 0.7
            )
            self.short_term_cache: Dict[str, ShortTermMemory] = {}
            # 用于存储待处理的摘要（异步回调中设置）
            self._pending_summary: Dict[str, str] = {}
            logger.info("记忆系统已启用")
        else:
            self.long_term = None
            self.summarizer = None
            self.importance_scorer = None
            self.short_term_cache = {}
            self._pending_summary = {}
            logger.info("记忆系统未启用（可通过 .env 开启）")

    # ============================================================
    # 短期记忆辅助方法
    # ============================================================
    def _get_or_create_short_term(self, user_id: str) -> Optional[ShortTermMemory]:
        """获取或创建用户的短期记忆窗口"""
        if not self.enable_memory:
            return None

        if user_id not in self.short_term_cache:
            logger.info(f"[短期记忆] 首次创建窗口，user_id={user_id}")
            self.short_term_cache[user_id] = ShortTermMemory(
                window_size=config.memory_window_size if hasattr(config, 'memory_window_size') else 10,
                on_overflow=lambda msgs: self._on_window_overflow(user_id, msgs),
            )
        return self.short_term_cache[user_id]

    def _on_window_overflow(self, user_id: str, old_messages: List[Dict[str, Any]]) -> None:
        """窗口满时的回调：触发摘要压缩"""
        if not self.enable_memory:
            return
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._async_summarize(user_id, old_messages))
        except RuntimeError:
            asyncio.create_task(self._async_summarize(user_id, old_messages))

    async def _async_summarize(self, user_id: str, old_messages: List[Dict[str, Any]]) -> None:
        """异步摘要压缩，并把摘要放回窗口"""
        if not self.enable_memory or self.summarizer is None:
            return
        
        # 消息太少，不值得压缩
        min_messages = 4
        if len(old_messages) < min_messages:
            return
            
        try:
            # 取最旧的消息进行压缩（排除最后一条，因为可能刚添加）
            messages_to_summarize = old_messages[:-1] if len(old_messages) > 1 else old_messages
            
            if len(messages_to_summarize) < min_messages:
                return
                
            summary = await self.summarizer.summarize(messages_to_summarize)
            if summary:
                logger.info(f"[摘要压缩] 完成: {summary[:50]}...")
                
                # 把摘要放回短期记忆窗口
                short_term = self._get_or_create_short_term(user_id)
                if short_term:
                    short_term.add("system", f"历史摘要：{summary}")
                    logger.debug(f"[摘要压缩] 摘要已放入窗口，当前窗口大小: {short_term.size()}")
                
        except Exception as e:
            logger.error(f"摘要压缩失败: {e}")

    def _format_short_term_context(self, user_id: str, max_messages: int = 6) -> str:
        """从短期记忆窗口获取格式化的上下文"""
        if not self.enable_memory or not user_id:
            return ""
        
        short_term = self._get_or_create_short_term(user_id)
        if short_term is None:
            return ""
        
        context_messages = short_term.get_context()
        logger.info(f"[短期记忆] 获取到 {len(context_messages)} 条消息: {context_messages}")
        
        if not context_messages:
            logger.warning("[短期记忆] 窗口为空！")
            return ""
        
        # 只取最近的 max_messages 条
        recent_messages = context_messages[-max_messages:]
        
        # 格式化
        formatted = []
        for msg in recent_messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if role == 'system':
                formatted.append(f"[系统摘要]: {content}")
            else:
                formatted.append(f"[{role}]: {content}")
        
        result = "\n".join(formatted)
        logger.info(f"[短期记忆] 格式化后的上下文: {result[:200]}...")
        return result

    # ========== 对外暴露的短期记忆记录方法（由 rag_agent_service 调用） ==========
    def _record_user_question(self, user_id: str, question: str) -> None:
        """只记录用户问题到短期记忆"""
        if not self.enable_memory or not user_id:
            return
        short_term = self._get_or_create_short_term(user_id)
        if short_term is None:
            return
        short_term.add("user", question)
        logger.debug(f"[短期记忆] 记录用户问题，当前窗口大小: {short_term.size()}")

    def _record_assistant_answer(self, user_id: str, answer: str) -> None:
        """只记录助手回答到短期记忆"""
        if not self.enable_memory or not user_id:
            return
        short_term = self._get_or_create_short_term(user_id)
        if short_term is None:
            return
        short_term.add("assistant", answer)
        logger.debug(f"[短期记忆] 记录助手回答，当前窗口大小: {short_term.size()}")

    # 保留组合方法（供外部一次性记录，但内部不再使用）
    def _record_conversation(self, user_id: str, question: str, answer: str) -> None:
        """完整记录一轮对话（组合调用）"""
        self._record_user_question(user_id, question)
        self._record_assistant_answer(user_id, answer)

    # ============================================================
    # 长期记忆辅助方法
    # ============================================================
    async def _query_memory_context(self, question: str, user_id: str) -> str:
        """查询长期记忆并返回格式化的上下文"""
        if not self.enable_memory or self.long_term is None:
            return ""

        try:
            memory_count = config.memory_retrieval_top_k if hasattr(config, 'memory_retrieval_top_k') else 3
            context = await self.long_term.retrieve_as_context(
                question,
                user_id,
                top_k=memory_count,
                enable_time_decay=True
            )
            if context:
                logger.info(f"[记忆] 从长期记忆召回相关事实")
            return context
        except Exception as e:
            logger.warning(f"[记忆] 查询失败: {e}")
            return ""

    async def _add_to_memory_if_important(
        self,
        question: str,
        answer: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """判断问题或答案是否重要，若重要则存入长期记忆"""
        if not self.enable_memory or self.importance_scorer is None or self.long_term is None:
            logger.debug("[记忆] 记忆未启用或组件未初始化")
            return

        try:
            # ====== 检查问题是否重要 ======
            importance_result = await self.importance_scorer.score(question)
            logger.info(f"[记忆] 问题重要性分数: {importance_result.score:.2f}, 阈值: {self.importance_scorer.threshold}")
            
            if importance_result.score >= self.importance_scorer.threshold:
                for fact in importance_result.facts:
                    await self.long_term.add_fact(
                        content=fact,
                        user_id=user_id,
                        importance=importance_result.score,
                        metadata={
                            "type": "question",
                            "metadata": metadata or {},
                        }
                    )
                logger.info(f"[记忆] ✅ 存入重要问题: {question[:30]}... (分数: {importance_result.score:.2f})")
            else:
                logger.debug(f"[记忆] ⏭️ 跳过存入问题（分数 {importance_result.score:.2f} < 阈值 {self.importance_scorer.threshold}）")

            # ====== 检查答案是否重要 ======
            answer_importance = await self.importance_scorer.score(answer)
            if answer_importance.score >= self.importance_scorer.threshold:
                for fact in answer_importance.facts:
                    await self.long_term.add_fact(
                        content=fact,
                        user_id=user_id,
                        importance=answer_importance.score,
                        metadata={
                            "type": "answer",
                            "metadata": metadata or {},
                        }
                    )
                logger.info(f"[记忆] ✅ 存入重要答案片段: {answer[:30]}... (分数: {answer_importance.score:.2f})")

        except Exception as e:
            logger.warning(f"[记忆] 存入失败: {e}")

    # ============================================================
    # MCP 工具辅助方法
    # ============================================================
    async def _get_mcp_client(self):
        if self._mcp_client is None:
            from app.agent.mcp_client import get_mcp_client_with_retry
            self._mcp_client = await get_mcp_client_with_retry()
        return self._mcp_client

    async def _call_mcp_tool(self, tool_name: str, arguments: dict):
        client = await self._get_mcp_client()
        tools = await client.get_tools()

        target_tool = None
        for tool in tools:
            if tool.name == tool_name:
                target_tool = tool
                break

        if target_tool is None:
            logger.warning(f"未找到 MCP 工具: {tool_name}")
            return None

        try:
            result = await target_tool.ainvoke(arguments)
            logger.info(f"MCP 工具 {tool_name} 调用成功")
            return result
        except Exception as e:
            logger.error(f"MCP 工具 {tool_name} 调用失败: {e}")
            return None

    # ============================================================
    # 1. 闲聊处理器
    # ============================================================
    async def handle_chat(self, question: str, user_id: Optional[str] = None) -> str:
        logger.info(f"[闲聊直出] {question[:30]}...")

        from langchain_core.prompts import ChatPromptTemplate

        # ========== 获取短期记忆上下文 ==========
        short_term_context = self._format_short_term_context(user_id) if user_id else ""
        logger.info(f"[短期记忆] 闲聊场景 - 上下文: {short_term_context[:100] if short_term_context else '(空)'}")

        # ========== 查询长期记忆 ==========
        memory_context = ""
        if user_id:
            memory_context = await self._query_memory_context(question, user_id)

        # ========== 构建 Prompt ==========
        if memory_context:
            prompt_template = """
你是一个友好、专业的运维助手。

【关于用户的历史记录】
{memory_context}

【对话历史】
{short_term_context}

请自然地回复用户的消息。

用户说：{question}

回复要求：
- 保持友好、热情的语气
- 如果是问候，礼貌回应并询问是否需要帮助
- 如果是感谢，表示乐意继续提供帮助
- 回答简洁自然，不要过于冗长
"""
        else:
            prompt_template = """
你是一个友好、专业的运维助手。

【对话历史】
{short_term_context}

请自然地回复用户的消息。

用户说：{question}

回复要求：
- 保持友好、热情的语气
- 如果是问候，礼貌回应并询问是否需要帮助
- 如果是感谢，表示乐意继续提供帮助
- 回答简洁自然，不要过于冗长
"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "short_term_context": short_term_context or "无",
            "memory_context": memory_context or "无相关历史记录",
            "question": question
        })

        # 注意：不再在此处记录短期记忆，由上层 rag_agent_service 统一记录
        return response.content

    # ============================================================
    # 2. 知识查询处理器（支持记忆读写）
    # ============================================================
    async def handle_knowledge(self, question: str, user_id: Optional[str] = None, top_k: int = 3) -> str:
        """知识查询走快速 RAG 链路 + 记忆读写"""
        logger.info(f"[知识快速RAG] {question[:30]}...")

        from langchain_core.prompts import ChatPromptTemplate

        # ========== 获取短期记忆上下文 ==========
        short_term_context = self._format_short_term_context(user_id) if user_id else ""
        logger.info(f"[短期记忆] 知识查询场景 - 上下文: {short_term_context[:100] if short_term_context else '(空)'}")

        # ========== 查询长期记忆 ==========
        memory_context = ""
        if user_id:
            memory_context = await self._query_memory_context(question, user_id)

        # ========== RAG 检索 ==========
        docs = hybrid_retriever.retrieve(question, top_k=top_k)

        if not docs:
            # 如果没有检索到文档，但可能有记忆内容
            if memory_context:
                prompt = ChatPromptTemplate.from_template("""
你是一个专业的运维知识助手。

【对话历史】
{short_term_context}

【用户历史记录】
{memory_context}

【用户问题】
{question}

请基于用户的历史记录回答，如果历史记录不足以回答，请明确说明。
""")
                chain = prompt | self.llm
                response = await chain.ainvoke({
                    "short_term_context": short_term_context or "无",
                    "memory_context": memory_context,
                    "question": question
                })
                # 注意：不再在此处记录短期记忆，由上层统一记录
                # 尝试将回答存入长期记忆
                if user_id:
                    await self._add_to_memory_if_important(question, response.content, user_id, {"type": "knowledge"})
                return response.content
            else:
                logger.warning(f"[知识快速RAG] 未检索到相关文档且无记忆")
                return "未检索到相关知识，请尝试换一种问法或联系技术支持。"

        # 构建上下文
        context_parts = []
        for idx, doc in enumerate(docs, 1):
            source = doc.metadata.get("_source", "未知来源")
            content = doc.page_content
            context_parts.append(f"【文档{idx}】（来源：{source}）\n{content}")
        rag_context = "\n\n".join(context_parts)

        # ========== 构建 Prompt ==========
        if memory_context:
            prompt_template = """
你是一个专业的运维知识助手。请基于以下参考资料和用户的历史记录回答用户的问题。

【对话历史】
{short_term_context}

【用户历史记录】
{memory_context}

【参考资料】
{context}

【用户问题】
{question}

回答要求：
1. 结合用户历史记录和参考资料回答
2. 基于参考资料，不要编造信息
3. 如果参考资料不足以回答，请明确说明
4. 回答要清晰、有条理、重点突出
5. 如果涉及操作步骤，请分点列出
"""
        else:
            prompt_template = """
你是一个专业的运维知识助手。请基于以下参考资料回答用户的问题。

【对话历史】
{short_term_context}

【参考资料】
{context}

【用户问题】
{question}

回答要求：
1. 基于参考资料回答，不要编造信息
2. 如果参考资料不足以回答，请明确说明
3. 回答要清晰、有条理、重点突出
4. 如果涉及操作步骤，请分点列出
"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "short_term_context": short_term_context or "无",
            "memory_context": memory_context or "无相关历史记录",
            "context": rag_context,
            "question": question
        })

        # 不再在此处记录短期记忆，由上层统一记录
        # 存入长期记忆
        if user_id:
            await self._add_to_memory_if_important(question, response.content, user_id, {"type": "knowledge"})

        logger.info(f"[知识快速RAG] 完成，检索到 {len(docs)} 篇文档")
        return response.content

    # ============================================================
    # 3. 故障排查处理器
    # ============================================================
    async def handle_troubleshoot(self, question: str, user_id: str) -> str:
        logger.info(f"[故障排查] {question[:30]}...")

        from langchain_core.prompts import ChatPromptTemplate

        # ========== 获取短期记忆上下文 ==========
        short_term_context = self._format_short_term_context(user_id) if user_id else ""
        logger.info(f"[短期记忆] 故障排查场景 - 上下文: {short_term_context[:100] if short_term_context else '(空)'}")

        # ========== 查询长期记忆 ==========
        memory_context = await self._query_memory_context(question, user_id)

        # ========== 提取服务名 ==========
        service_name = self._extract_service_name(question)
        logger.info(f"[故障排查] 提取服务名: {service_name}")

        # ========== 调用日志工具 ==========
        log_context = ""
        try:
            topic_result = await self._call_mcp_tool(
                "search_topic_by_service_name",
                {"service_name": service_name, "fuzzy": True}
            )
            topic_id = None
            if topic_result and isinstance(topic_result, dict):
                topics = topic_result.get("topics", [])
                if topics:
                    topic_id = topics[0].get("topic_id")
                    logger.info(f"[故障排查] 找到 topic: {topic_id}")

            if topic_id:
                timestamp_result = await self._call_mcp_tool("get_current_timestamp", {})
                if timestamp_result:
                    current_ts = int(timestamp_result) if isinstance(timestamp_result, int) else int(timestamp_result)
                    start_ts = current_ts - (30 * 60 * 1000)
                    log_result = await self._call_mcp_tool(
                        "search_log",
                        {
                            "topic_id": topic_id,
                            "start_time": start_ts,
                            "end_time": current_ts,
                            "limit": 50
                        }
                    )
                    if log_result and isinstance(log_result, dict):
                        logs = log_result.get("logs", [])
                        if logs:
                            log_context = "查询到以下日志记录：\n"
                            for log in logs[:10]:
                                log_context += f"- {log.get('timestamp', '')} [{log.get('level', 'INFO')}] {log.get('message', '')}\n"
                            logger.info(f"[故障排查] 获取到 {len(logs)} 条日志")
                        else:
                            log_context = "未查询到相关日志"
                    else:
                        log_context = f"未找到服务 '{service_name}' 对应的日志主题"
        except Exception as e:
            logger.warning(f"[故障排查] CLS 日志查询失败: {e}")
            log_context = f"日志查询失败: {str(e)}"

        # ========== 调用监控工具 ==========
        monitor_context = ""
        try:
            cpu_result = await self._call_mcp_tool(
                "query_cpu_metrics",
                {"service_name": service_name, "interval": "1m"}
            )
            if cpu_result and isinstance(cpu_result, dict):
                stats = cpu_result.get("statistics", {})
                alert = cpu_result.get("alert_info", {})
                monitor_context += f"CPU 使用率统计:\n"
                monitor_context += f"  - 平均值: {stats.get('avg', 'N/A')}%\n"
                monitor_context += f"  - 最大值: {stats.get('max', 'N/A')}%\n"
                monitor_context += f"  - 告警状态: {alert.get('message', '正常')}\n"

            memory_result = await self._call_mcp_tool(
                "query_memory_metrics",
                {"service_name": service_name, "interval": "1m"}
            )
            if memory_result and isinstance(memory_result, dict):
                stats = memory_result.get("statistics", {})
                alert = memory_result.get("alert_info", {})
                monitor_context += f"内存使用率统计:\n"
                monitor_context += f"  - 平均值: {stats.get('avg', 'N/A')}%\n"
                monitor_context += f"  - 最大值: {stats.get('max', 'N/A')}%\n"
                monitor_context += f"  - 告警状态: {alert.get('message', '正常')}\n"
        except Exception as e:
            logger.warning(f"[故障排查] 监控查询失败: {e}")
            monitor_context = f"监控数据查询失败: {str(e)}"

        # ========== RAG 检索 ==========
        docs = hybrid_retriever.retrieve(question, top_k=3)
        doc_context = "\n\n".join([doc.page_content for doc in docs]) if docs else "未找到相关技术文档"

        # ========== LLM 综合回答 ==========
        prompt = ChatPromptTemplate.from_template("""
你是一个经验丰富的运维专家。请基于以下信息进行故障排查和诊断。

【对话历史】
{short_term_context}

【用户历史记录】
{memory_context}

【日志信息】
{log_context}

【监控信息】
{monitor_context}

【技术文档参考】
{doc_context}

【用户问题】
{question}

请按以下结构回答：
1. **问题分析**：根据日志和监控信息，判断可能的故障原因
2. **排查步骤**：给出具体的排查命令和检查点
3. **解决方案**：提供可操作的修复步骤
4. **预防措施**：如何避免再次发生

如果某些信息缺失，请基于已有信息给出最佳判断，并说明假设条件。
""")

        chain = prompt | self.llm
        response = await chain.ainvoke({
            "short_term_context": short_term_context or "无",
            "memory_context": memory_context or "无相关历史记录",
            "log_context": log_context,
            "monitor_context": monitor_context,
            "doc_context": doc_context,
            "question": question
        })

        # 不再在此处记录短期记忆，由上层统一记录
        # 存入长期记忆
        await self._add_to_memory_if_important(question, response.content, user_id, {"type": "troubleshoot"})

        logger.info(f"[故障排查] 完成")
        return response.content

    def _extract_service_name(self, question: str) -> str:
        service_keywords = [
            "nginx", "redis", "mysql", "kafka", "elasticsearch",
            "data-sync", "api-gateway", "monitor", "cls"
        ]
        question_lower = question.lower()
        for keyword in service_keywords:
            if keyword in question_lower:
                return keyword
        return "data-sync-service"

    # ============================================================
    # 4. 系统控制器
    # ============================================================
    async def handle_control(self, question: str, user_id: str) -> str:
        logger.info(f"[系统控制] {question[:30]}...")

        from langchain_core.prompts import ChatPromptTemplate

        # ========== 获取短期记忆上下文 ==========
        short_term_context = self._format_short_term_context(user_id) if user_id else ""

        # ========== 查询长期记忆 ==========
        memory_context = await self._query_memory_context(question, user_id)

        try:
            result = await self._call_mcp_tool("execute_control", {"query": question})
            if result:
                return f"操作已执行：\n{result}"
        except Exception as e:
            logger.warning(f"[系统控制] 控制工具不可用，进入模拟模式: {e}")

        prompt = ChatPromptTemplate.from_template("""
你是一个运维助手。用户要求执行以下操作：

【对话历史】
{short_term_context}

【用户历史记录】
{memory_context}

用户请求：{question}

请分析用户想要执行的具体操作（重启/扩容/限流/其他），并以清晰的格式返回。

当前模式：模拟执行（未连接实际 K8s 集群）

输出格式：
【操作类型】重启/扩容/限流/其他
【目标资源】具体服务或实例
【执行结果】模拟执行成功/失败
【后续建议】相关注意事项
""")

        chain = prompt | self.llm
        response = await chain.ainvoke({
            "short_term_context": short_term_context or "无",
            "memory_context": memory_context or "无相关历史记录",
            "question": question
        })

        # 不再在此处记录短期记忆，由上层统一记录
        # 存入长期记忆
        await self._add_to_memory_if_important(question, response.content, user_id, {"type": "control"})

        logger.info(f"[系统控制] 模拟执行完成")
        return response.content

    # ============================================================
    # 5. 兜底处理器
    # ============================================================
    async def handle_unknown(self, question: str, user_id: Optional[str] = None) -> str:
        """未知意图走知识查询作为兜底，并尝试记忆写入"""
        logger.info(f"[兜底] {question[:30]}...")
        result = await self.handle_knowledge(question, user_id)
        return result


# 全局单例
route_handlers = RouteHandlers()