"""plana test：网络与各依赖服务连通性检查。"""
import asyncio
import sys
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from plana_chat.config import load_config, env


async def main() -> None:
    cfg = load_config()
    proxy = cfg.telegram.get("proxy", "")
    token = env(cfg.telegram.token_env)

    # Telegram 走代理；DeepSeek / Ollama 直连（与真实 bot 一致）
    tg_kwargs: dict = {"timeout": 15, "trust_env": False}
    if proxy:
        tg_kwargs["proxy"] = proxy
    direct_kwargs: dict = {"timeout": 15, "trust_env": False}

    async with httpx.AsyncClient(**tg_kwargs) as tg, \
            httpx.AsyncClient(**direct_kwargs) as direct:
        # 1. Telegram API（走代理）
        print("1. Telegram API（走代理）")
        if not token:
            print("   ❌ 未配置 TELEGRAM_BOT_TOKEN")
        else:
            try:
                r = await tg.get(f"https://api.telegram.org/bot{token}/getMe")
                r.raise_for_status()
                d = r.json()
                if d.get("ok"):
                    print(f"   ✅ 连接成功，bot @{d['result']['username']}")
                else:
                    print(f"   ❌ {d.get('description')}")
            except Exception as e:
                print(f"   ❌ {type(e).__name__}: {e}")

        # 2. 对话引擎（DeepSeek / Ollama，直连）
        provider = cfg.llm.get("provider", "deepseek")
        print(f"2. 对话引擎（{provider}，直连）")
        if provider == "deepseek":
            try:
                key = env(cfg.llm.api_key_env)
                r = await direct.post(
                    f"{cfg.llm.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": cfg.llm.model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 5,
                    },
                )
                r.raise_for_status()
                print("   ✅ 返回正常")
            except Exception as e:
                print(f"   ❌ {type(e).__name__}: {e}")
        else:
            try:
                r = await direct.get("http://127.0.0.1:11434/api/tags", timeout=5)
                r.raise_for_status()
                print("   ✅ Ollama 在运行")
            except Exception as e:
                print(f"   ❌ {type(e).__name__}: {e}")

        # 3. 本地 Ollama（embedding 模型，直连）
        print("3. 本地 Ollama（embedding，直连）")
        try:
            r = await direct.get("http://127.0.0.1:11434/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"   ✅ 在运行（模型：{', '.join(models) or '无'}）")
        except Exception as e:
            print(f"   ❌ {type(e).__name__}: {e}")

    # 4. TTS 语音引擎
    print("4. TTS 语音引擎")
    sock = Path(cfg.tts.get("socket", "/tmp/plana_tts.sock"))
    if sock.exists():
        print(f"   ✅ socket 存在（{sock}）")
    else:
        print("   ⚠️ socket 不存在（首次发语音时会自动启动）")


asyncio.run(main())
