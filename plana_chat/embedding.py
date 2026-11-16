"""本地文本 embedding（Ollama）。"""
from __future__ import annotations

import httpx


async def embed_text(
    text: str,
    host: str = "http://127.0.0.1:11434",
    model: str = "bge-m3",
    timeout: float = 60,
) -> list[float]:
    """调用 Ollama /api/embed，返回文本的向量（bge-m3，中文/多语种友好）。"""
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        resp = await client.post(
            f"{host}/api/embed",
            json={"model": model, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
