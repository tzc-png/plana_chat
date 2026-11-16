"""语音格式转换：通道无关的 WAV 字节 → 各通道格式。"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def wav_to_ogg_opus(wav_bytes: bytes, bitrate: str = "64k") -> bytes:
    """WAV 字节 → OGG(Opus) 字节，Telegram 语音消息格式。"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
        f.write(wav_bytes)

    ogg_path = tempfile.mktemp(suffix=".ogg")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", wav_path,
                "-c:a", "libopus", "-b:a", bitrate,
                ogg_path,
            ],
            check=True,
        )
        return Path(ogg_path).read_bytes()
    finally:
        Path(wav_path).unlink(missing_ok=True)
        Path(ogg_path).unlink(missing_ok=True)
