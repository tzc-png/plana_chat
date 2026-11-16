"""本地 Ollama 客户端（/api/chat，messages 格式）。"""
from __future__ import annotations

import httpx

from .base import ChatLLM


class OllamaLLM(ChatLLM):
    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5:14b",
        temperature: float = 0.85,
        num_ctx: int = 4096,
        timeout: float = 300,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout

    async def chat(self, messages: list[dict]) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            resp = await client.post(f"{self.host}/api/chat", json=body)
            resp.raise_for_status()
            data = resp.json()

        return data["message"]["content"]
