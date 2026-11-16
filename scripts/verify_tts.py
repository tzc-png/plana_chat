"""独立验证 TTS 链路。

用法：
    python scripts/verify_tts.py                # 完整链路：中文→翻译→日语→TTS
    python scripts/verify_tts.py "おはよう"       # 直接合成给定日语文本
"""
import asyncio
import sys
import tempfile
import wave
from pathlib import Path

from plana_chat.config import load_config
from plana_chat.translator import Translator
from plana_chat.tts import TTSServerManager, synthesize


async def main() -> None:
    cfg = load_config()

    if len(sys.argv) > 1:
        ja_text = sys.argv[1]
        print(f"[直接合成] {ja_text}")
    else:
        tr = Translator(cfg)
        ja_text = await tr.translate("老师，早上好。今天也要加油哦。")
        print(f"[翻译结果] {ja_text}")

    mgr = TTSServerManager(cfg)
    print("[TTS] 启动/连接语音引擎...")
    mgr.ensure_running()

    out = tempfile.mktemp(suffix=".wav")
    synthesize(ja_text, out, cfg.tts.socket, cfg.tts.lang)

    with wave.open(out, "rb") as w:
        rate = w.getframerate()
        dur = w.getnframes() / rate
    size = Path(out).stat().st_size
    print(f"[OK] {out}\n     采样率={rate}Hz  时长={dur:.2f}s  大小={size} 字节")
    print("可用 aplay %s 试听" % out)


if __name__ == "__main__":
    asyncio.run(main())
