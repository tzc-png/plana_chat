"""管理 plana_tts_server.py 子进程的生命周期（启动/保活）。"""
from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

from ..config import Config


class TTSServerManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _sock_exists(self) -> bool:
        return Path(self.cfg.tts.socket).exists()

    def _probe(self) -> bool:
        """连接 socket 看服务是否真的活着（排除残留 socket 文件）。"""
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect(self.cfg.tts.socket)
            s.close()
            return True
        except OSError:
            return False

    def is_running(self) -> bool:
        return self._sock_exists() and self._probe()

    def ensure_running(self) -> None:
        if self.is_running():
            return

        log_path = Path(self.cfg.tts.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "ab")

        env = os.environ.copy()
        env.update(
            {
                "PLANA_ENGINE_DIR": self.cfg.tts.engine_dir,
                "PLANA_GPT_PATH": self.cfg.tts.gpt_path,
                "PLANA_VITS_PATH": self.cfg.tts.vits_path,
                "PLANA_REF_WAV": self.cfg.tts.ref_wav,
                "PLANA_REF_TXT": self.cfg.tts.ref_txt,
                "PLANA_SOCK": self.cfg.tts.socket,
            }
        )

        subprocess.Popen(
            [self.cfg.tts.python, self.cfg.tts.server_script],
            env=env,
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
        )

        # 模型加载较慢，最长等 180s
        for _ in range(180):
            if self.is_running():
                return
            time.sleep(1)
        raise TimeoutError("TTS 服务启动超时（180s），请查看 data/tts.log")

    def stop(self) -> None:
        """停止 TTS 服务（按 socket 关联进程终止不可靠，这里直接清理 socket 由外部管理）。"""
        # shittim 用 pkill -f plana_tts_server.py；这里保持最小实现，仅供测试脚本使用
        pass
