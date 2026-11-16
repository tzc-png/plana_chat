"""对话引擎抽象与工厂。"""
from __future__ import annotations

from ..config import Config, env
from .base import ChatLLM
from .ollama import OllamaLLM
from .openai_compat import OpenAICompatLLM


def build_llm(cfg: Config) -> ChatLLM:
    """按配置构建对话 LLM。"""
    if cfg.llm.provider == "ollama":
        return OllamaLLM(
            host=cfg.ollama.host,
            model=cfg.ollama.model,
            temperature=cfg.ollama.get("temperature", 0.85),
            num_ctx=cfg.ollama.get("num_ctx", 4096),
            timeout=cfg.ollama.get("timeout", 300),
        )
    # 默认 OpenAI 兼容（DeepSeek 等）
    api_key = env(cfg.llm.api_key_env)
    return OpenAICompatLLM(
        base_url=cfg.llm.base_url,
        api_key=api_key,
        model=cfg.llm.model,
        temperature=cfg.llm.get("temperature", 0.85),
        timeout=cfg.llm.get("timeout", 120),
        extra_body=cfg.llm.get("extra_body", {}).as_dict(),
    )


def build_local_llm(cfg: Config) -> ChatLLM:
    """构建本地 Ollama 降级实例。"""
    return OllamaLLM(
        host=cfg.ollama.host,
        model=cfg.ollama.model,
        temperature=cfg.ollama.get("temperature", 0.85),
        num_ctx=cfg.ollama.get("num_ctx", 4096),
        timeout=cfg.ollama.get("timeout", 300),
    )


__all__ = ["ChatLLM", "OpenAICompatLLM", "OllamaLLM", "build_llm", "build_local_llm"]
