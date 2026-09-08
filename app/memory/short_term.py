"""短期记忆 - 滑动窗口

特点：
1. 保留最近 N 轮对话
2. 窗口满时触发摘要压缩回调
3. 纯内存操作，极快
"""

from collections import deque
from typing import List, Dict, Any, Optional, Callable
from loguru import logger


class ShortTermMemory:
    """滑动窗口短期记忆"""

    def __init__(
        self,
        window_size: int = 10,
        on_overflow: Optional[Callable] = None,
    ):
        """
        Args:
            window_size: 窗口大小（消息条数，1轮=2条：user+assistant）
            on_overflow: 窗口满时的回调函数（用于触发摘要压缩）
        """
        self.window_size = window_size
        self.window: deque = deque(maxlen=window_size)
        self.on_overflow = on_overflow
        self._overflow_triggered = False

    def add(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """
        添加一条消息到窗口

        Args:
            role: user / assistant / system
            content: 消息内容
            metadata: 额外信息（时间戳、意图等）
        """
        # 如果窗口已满，触发回调（在添加新消息之前）
        if len(self.window) >= self.window_size and self.on_overflow and not self._overflow_triggered:
            logger.debug(f"窗口已满（{len(self.window)}/{self.window_size}），触发摘要压缩回调")
            self._overflow_triggered = True
            self.on_overflow(list(self.window))
            # 回调应该清空窗口并放入摘要，重置标志
            self._overflow_triggered = False

        message = {"role": role, "content": content, "metadata": metadata or {}}
        self.window.append(message)

    def get_context(self) -> List[Dict[str, Any]]:
        """获取当前窗口中的所有消息（用于拼接到 Prompt）"""
        return list(self.window)

    def clear(self) -> None:
        """清空窗口"""
        self.window.clear()
        self._overflow_triggered = False

    def is_full(self) -> bool:
        """检查窗口是否已满"""
        return len(self.window) >= self.window_size

    def size(self) -> int:
        """当前消息数量"""
        return len(self.window)

    def get_last_n(self, n: int) -> List[Dict[str, Any]]:
        """获取最近 N 条消息"""
        return list(self.window)[-n:]

    def get_first_n(self, n: int) -> List[Dict[str, Any]]:
        """获取前 N 条消息（用于摘要压缩）"""
        return list(self.window)[:n]