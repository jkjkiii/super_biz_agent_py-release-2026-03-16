"""重要性阈值打分器

判断一条消息是否值得存入长期记忆
"""

import re
from typing import Dict, Any, Optional, List
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from pydantic import BaseModel

from app.config import config


class ImportanceResult(BaseModel):
    """重要性打分结果"""
    score: float  # 0-1
    reason: str
    facts: List[str]  # 提取的事实列表


class ImportanceScorer:
    """
    重要性阈值打分器

    支持两种模式：
    1. 规则模式（快速，覆盖常见场景）
    2. LLM 模式（准确，处理复杂情况）
    """

    def __init__(self, threshold: float = 0.7, use_llm: bool = True):
        self.threshold = threshold
        self.use_llm = use_llm

        if use_llm:
            self.llm = ChatQwen(
                model=config.rag_model,
                api_key=config.dashscope_api_key,
                temperature=0.1,
            )
            self.prompt = ChatPromptTemplate.from_template("""
你是一个信息价值评估专家。请判断以下对话内容是否值得永久记住。

**值得记住的标准（高分 0.8+）**：
- 包含具体错误码（如 500213, ConnectionTimeout）
- 包含技术配置（如 IP 地址、端口、环境变量、版本号）
- 包含解决方案或操作步骤
- 用户明确表达偏好（如“我喜欢用 Python”）
- 关键服务名、实例名

**不值得记住的标准（低分 < 0.5）**：
- 日常问候（“你好”“谢谢”）
- 重复确认（“好的”“知道了”）
- 无关闲聊

**输入内容**：
{content}

**输出 JSON 格式**：
{{
    "score": 0.85,
    "reason": "包含错误码 500213",
    "facts": ["服务 data-sync-service 报错 500213"]
}}
""")

    def _rule_based_score(self, content: str) -> Dict[str, Any]:
        """基于规则的重要性打分（快速路径）"""
        score = 0.0
        reasons = []
        facts = []

        content_lower = content.lower()

        # ====== 1. IP 地址检测（高权重） ======
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        if re.search(ip_pattern, content_lower):
            score += 0.35   # 直接加 0.35！
            reasons.append("包含 IP 地址")
            # 提取 IP 作为事实
            ips = re.findall(ip_pattern, content)
            for ip in ips:
                facts.append(f"服务器 IP: {ip}")

        # ====== 2. 端口号检测 ======
        port_pattern = r'port\s*[:=]?\s*(\d{4,5})'
        if re.search(port_pattern, content_lower):
            score += 0.25
            reasons.append("包含端口号")
            ports = re.findall(port_pattern, content)
            for port in ports:
                facts.append(f"端口: {port}")

        # ====== 3. 错误码检测 ======
        error_patterns = [
            r'\b\d{4,}\b',      # 4位以上数字
            r'error\s*code',
            r'timeout',
            r'failed',
            r'exception',
            r'oom',
        ]
        for pattern in error_patterns:
            if re.search(pattern, content_lower):
                score += 0.15
                reasons.append(f"包含错误关键词")
                facts.append(f"错误信息: {content[:50]}")

        # ====== 4. 配置信息检测 ======
        config_keywords = ['配置', '设置', '版本', '环境', '变量', '.env', '.yaml']
        for kw in config_keywords:
            if kw in content:
                score += 0.1
                reasons.append(f"包含配置关键词: {kw}")

        # ====== 5. 服务名检测 ======
        service_keywords = [
            'nginx', 'redis', 'mysql', 'kafka', 'elasticsearch',
            'service', 'pod', 'container', 'k8s', 'kubernetes',
            'data-sync', 'api-gateway'
        ]
        for kw in service_keywords:
            if kw in content_lower:
                score += 0.1
                reasons.append(f"包含服务关键词: {kw}")

        # ====== 6. 具体数值检测（如配置值） ======
        # 检测 "是 192.168.1.100" 或 "= 8080" 这种赋值语句
        if re.search(r'[=:]\s*[\d\.]+', content) or re.search(r'是\s*[\d\.]+', content):
            score += 0.2
            reasons.append("包含具体数值/配置")

        # ====== 7. 长度加分 ======
        if len(content) > 20:
            score += 0.05

        # ====== 8. 闲聊关键词排除 ======
        chat_keywords = ['你好', '谢谢', '哈哈', '好的', 'ok', '嗯', '哦']
        for kw in chat_keywords:
            if kw in content_lower:
                score -= 0.15

        # 确保在 0-1 范围内
        score = max(0, min(1, score))

        # 如果没有提取到事实，但分数较高，提取完整内容作为事实
        if not facts and score > 0.5:
            facts = [content[:100]]

        logger.debug(f"规则打分: {score:.2f}, 原因: {reasons[:3]}")

        return {
            "score": score,
            "reason": ", ".join(reasons[:3]) if reasons else "规则未匹配到关键特征",
            "facts": facts if facts else [],
        }

    async def _llm_based_score(self, content: str) -> Dict[str, Any]:
        """基于 LLM 的重要性打分（慢速路径）"""
        try:
            chain = self.prompt | self.llm
            response = await chain.ainvoke({"content": content})

            # 尝试解析 JSON
            import json
            # 找到 JSON 部分
            text = response.content
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                result = json.loads(json_str)
                return {
                    "score": min(1.0, max(0.0, float(result.get("score", 0)))),
                    "reason": result.get("reason", "LLM 评估"),
                    "facts": result.get("facts", [content[:100]]),
                }
        except Exception as e:
            logger.warning(f"LLM 重要性打分失败: {e}")

        # 降级到规则模式
        return self._rule_based_score(content)

    async def score(self, content: str) -> ImportanceResult:
        """
        对一条消息进行重要性打分

        Args:
            content: 消息内容

        Returns:
            ImportanceResult: 包含分数、原因、提取的事实
        """
        if not content or not content.strip():
            return ImportanceResult(score=0.0, reason="内容为空", facts=[])

        # 先试规则模式（快速）
        rule_result = self._rule_based_score(content)

        # 如果规则分数很高或很低，直接返回
        if rule_result["score"] > 0.8 or rule_result["score"] < 0.2:
            logger.debug(f"规则打分结果: {rule_result['score']:.2f}, 原因: {rule_result['reason']}")
            return ImportanceResult(
                score=rule_result["score"],
                reason=rule_result["reason"],
                facts=rule_result["facts"],
            )

        # 中间分数用 LLM 精修
        if self.use_llm:
            logger.debug(f"规则分数 {rule_result['score']:.2f}，调用 LLM 精修")
            llm_result = await self._llm_based_score(content)
            # 取两种分数平均
            final_score = (rule_result["score"] + llm_result["score"]) / 2
            return ImportanceResult(
                score=final_score,
                reason=f"规则({rule_result['score']:.2f}) + LLM({llm_result['score']:.2f})",
                facts=llm_result["facts"],
            )

        return ImportanceResult(
            score=rule_result["score"],
            reason=rule_result["reason"],
            facts=rule_result["facts"],
        )

    async def is_important(self, content: str) -> bool:
        """快速判断是否重要（是否超过阈值）"""
        result = await self.score(content)
        return result.score >= self.threshold

    def is_important_sync(self, content: str) -> bool:
        """同步快速判断（仅用规则）"""
        result = self._rule_based_score(content)
        return result["score"] >= self.threshold