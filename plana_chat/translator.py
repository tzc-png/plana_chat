"""中日翻译：中文回复 → 日语（普拉娜音色仅支持日语）。"""
from __future__ import annotations

from .config import Config
from .llm.base import ChatLLM
from .llm.ollama import OllamaLLM
from .llm.openai_compat import OpenAICompatLLM


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
        else:
            from .config import env

            api_key = env(cfg.llm.api_key_env)
            self._llm = OpenAICompatLLM(
                base_url=cfg.llm.base_url,
                api_key=api_key,
                model=cfg.translator.model,
                temperature=cfg.translator.get("temperature", 0.3),
                timeout=cfg.llm.get("timeout", 120),
                extra_body=cfg.llm.get("extra_body", {}).as_dict(),
            )

    async def translate(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        result = await self._llm.chat(
            [{"role": "user", "content": f"{self.prompt}{text}"}]
        )
        return "".join(result.split())  # 去空白（含换行）
