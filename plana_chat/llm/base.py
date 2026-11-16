"""对话引擎统一接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ChatLLM(ABC):
    """对话模型抽象：输入 messages，输出助手文本。"""

    @abstractmethod
    async def chat(self, messages: list[dict]) -> str:
        """messages: [{"role": "system"|"user"|"assistant", "content": str}]"""
        raise NotImplementedError
