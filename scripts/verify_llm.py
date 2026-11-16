"""独立验证对话引擎：云端 DeepSeek + 本地 Ollama 各回一条人格对话。"""
import asyncio
from pathlib import Path

from plana_chat.config import load_config
from plana_chat.llm import build_llm, build_local_llm

ROOT = Path(__file__).resolve().parent.parent


async def main() -> None:
    cfg = load_config()
    persona = (
        (ROOT / "plana_chat" / "character" / "plana.txt")
        .read_text(encoding="utf-8")
        .replace("{current_time}", "2026-01-01 08:00:00")
    )
    messages = [
        {"role": "system", "content": persona},
        {"role": "user", "content": "老师，早上好。"},
    ]

    print("=== 主对话引擎（provider=%s）===" % cfg.llm.provider)
    llm = build_llm(cfg)
    try:
        reply = await llm.chat(messages)
        print(reply)
    except Exception as e:
        print(f"[失败] {type(e).__name__}: {e}")

    if cfg.llm.get("fallback_to_local", False):
        print("\n=== 本地 Ollama（降级）===")
        local = build_local_llm(cfg)
        try:
            reply = await local.chat(messages)
            print(reply)
        except Exception as e:
            print(f"[失败] {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
