"""TTS 客户端与子进程管理（复用 shittim 的 plana_tts_server.py）。"""
from .client import synthesize
from .manager import TTSServerManager

__all__ = ["synthesize", "TTSServerManager"]
