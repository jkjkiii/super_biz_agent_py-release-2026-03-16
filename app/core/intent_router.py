"""意图识别与动态路由模块"""

from typing import Optional, Literal, Dict, Any
from enum import Enum
from pydantic import BaseModel
from loguru import logger

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.config import config


class Intent(str, Enum):
    TROUBLESHOOT = "troubleshoot"
    KNOWLEDGE = "knowledge"
    CHAT = "chat"
    CONTROL = "control"
    UNKNOWN = "unknown"


class RouteResult(BaseModel):
    intent: Intent
    confidence: float
    reason: Optional[str] = None
    entities: Optional[Dict[str, Any]] = None


class IntentRouter:
    """意图路由器：关键词匹配（Fast Path）+ LLM 分类（Slow Path）"""
    
    def __init__(self):
        # 初始化 LLM（用于 Slow Path）
        self.llm = ChatTongyi(
            model="qwen-max",
            api_key=config.dashscope_api_key,
            top_p=0.1
        )
        
        self.parser = PydanticOutputParser(pydantic_object=RouteResult)
        
        self.keyword_map = self._build_keyword_map()
        
        self.prompt = ChatPromptTemplate.from_template("""
你是一个专业的意图识别专家。请分析用户输入的意图。

意图分类规则：
1. **troubleshoot（故障排查）**：
   - 报告异常：CPU高、内存满、磁盘不足、服务不可用、报错、超时、崩溃、卡顿、慢
   - 关键词：报错、失败、异常、超时、宕机、不可用、高、满、慢、重启（指异常重启）
   
2. **knowledge（知识查询）**：
   - 询问概念、原理、操作步骤、最佳实践
   - 关键词：是什么、怎么配置、如何优化、为什么、原理、步骤、方法、最佳实践

3. **chat（闲聊）**：
   - 日常问候、表达情绪、无实质内容
   - 关键词：你好、谢谢、哈哈、怎么样、天气

4. **control（系统控制）**：
   - 要求执行操作：重启、扩容、限流、开启、关闭
   - 关键词：帮我重启、扩容、限流、开启、关闭、执行、操作

用户输入：{user_input}

{format_instructions}
""")
        
        self.chain = self.prompt | self.llm | self.parser
    
    def _build_keyword_map(self) -> Dict[str, Intent]:
        return {
            # ====== 已有故障排查关键词 ======
            "cpu": Intent.TROUBLESHOOT,
            "内存": Intent.TROUBLESHOOT,
            "磁盘": Intent.TROUBLESHOOT,
            "报错": Intent.TROUBLESHOOT,
            "错误": Intent.TROUBLESHOOT,
            "失败": Intent.TROUBLESHOOT,
            "超时": Intent.TROUBLESHOOT,
            "宕机": Intent.TROUBLESHOOT,
            "不可用": Intent.TROUBLESHOOT,
            "崩溃": Intent.TROUBLESHOOT,
            "卡顿": Intent.TROUBLESHOOT,
            "慢": Intent.TROUBLESHOOT,
            "高": Intent.TROUBLESHOOT,
            "满": Intent.TROUBLESHOOT,

            # ====== 已有知识查询关键词 ======
            "是什么": Intent.KNOWLEDGE,
            "怎么配置": Intent.KNOWLEDGE,
            "如何优化": Intent.KNOWLEDGE,
            "为什么": Intent.KNOWLEDGE,
            "原理": Intent.KNOWLEDGE,
            "步骤": Intent.KNOWLEDGE,
            "方法": Intent.KNOWLEDGE,
            "最佳实践": Intent.KNOWLEDGE,
            "区别": Intent.KNOWLEDGE,

            # ====== 🆕 新增：陈述句/提供信息的关键词 ======
            "我的": Intent.KNOWLEDGE,       # "我的服务器 IP 是..."
            "是": Intent.KNOWLEDGE,         # "IP 是 192.168.1.100"
            "IP": Intent.KNOWLEDGE,         # "服务器 IP 是..."
            "地址": Intent.KNOWLEDGE,       # "IP 地址"
            "配置": Intent.KNOWLEDGE,       # "配置了..."
            "设置": Intent.KNOWLEDGE,       # "设置成..."
            "版本": Intent.KNOWLEDGE,       # "版本是 v1.2"
            "端口": Intent.KNOWLEDGE,       # "端口是 8080"

            # ====== 已有控制操作关键词 ======
            "帮我重启": Intent.CONTROL,
            "扩容": Intent.CONTROL,
            "限流": Intent.CONTROL,
            "开启": Intent.CONTROL,
            "关闭": Intent.CONTROL,
            "执行": Intent.CONTROL,
            "操作": Intent.CONTROL,

            # ====== 已有闲聊关键词 ======
            "你好": Intent.CHAT,
            "谢谢": Intent.CHAT,
        }
    
    def _fast_match(self, query: str) -> Optional[RouteResult]:
        query_lower = query.lower()
        sorted_keywords = sorted(self.keyword_map.keys(), key=len, reverse=True)
        
        for keyword in sorted_keywords:
            if keyword in query_lower:
                intent = self.keyword_map[keyword]
                return RouteResult(
                    intent=intent,
                    confidence=0.85,
                    reason=f"关键词匹配: '{keyword}'"
                )
        return None
    
    def _llm_classify(self, query: str) -> RouteResult:
        try:
            result = self.chain.invoke({
                "user_input": query,
                "format_instructions": self.parser.get_format_instructions()
            })
            return result
        except Exception as e:
            logger.error(f"LLM 分类失败: {e}")
            return RouteResult(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                reason=f"LLM 分类失败: {str(e)}"
            )
    
    def route(self, query: str) -> RouteResult:
        # 1. Fast Path
        fast_result = self._fast_match(query)
        if fast_result is not None:
            return fast_result
        
        # 2. Slow Path
        logger.debug(f"关键词未匹配，调用 LLM 分类: {query[:30]}...")
        return self._llm_classify(query)


# 全局单例
intent_router = IntentRouter()