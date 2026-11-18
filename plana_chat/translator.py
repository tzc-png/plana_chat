"""中日翻译：中文回复 → 日语（普拉娜音色仅支持日语）。"""
from __future__ import annotations

import logging

from .config import Config, env
from .llm.base import ChatLLM
from .llm.ollama import OllamaLLM
from .llm.openai_compat import OpenAICompatLLM

logger = logging.getLogger("plana_chat.translator")


class Translator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prompt = cfg.translator.prompt
        provider = cfg.translator.get("provider", cfg.llm.provider)

        if provider == "ollama":
            self._llm: ChatLLM = OllamaLLM(
                host=cfg.ollama.host,
                model=cfg.translator.ollama_model,
                temperature=cfg.translator.get("temperature", 0.3),
                timeout=cfg.ollama.get("timeout", 300),
            )
            self._local_llm: ChatLLM | None = None  # 本身就是本地，无需降级
        else:
            self._llm = OpenAICompatLLM(
                base_url=cfg.llm.base_url,
                api_key=env(cfg.llm.api_key_env),
                model=cfg.translator.model,
                temperature=cfg.translator.get("temperature", 0.3),
                timeout=cfg.llm.get("timeout", 120),
                extra_body=cfg.llm.get("extra_body", {}).as_dict(),
            )
            self._local_llm = OllamaLLM(
                host=cfg.ollama.host,
                model=cfg.translator.ollama_model,
                temperature=cfg.translator.get("temperature", 0.3),
                timeout=cfg.ollama.get("timeout", 300),
            )

    async def translate(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        messages = [{"role": "user", "content": f"{self.prompt}{text}"}]
        try:
            result = await self._llm.chat(messages)
        except Exception as e:
            if self._local_llm is None:
                raise
            logger.warning(
                "\033[31m[降级] 翻译云端失败，已切换到本地 Ollama：%s\033[0m", e
            )
            result = await self._local_llm.chat(messages)
        result_clean = "".join(result.split())
        logger.debug("翻译请求：%s → %s", text[:40], result_clean[:40])
        return result_clean
