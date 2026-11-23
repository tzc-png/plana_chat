"""记忆编辑互斥：bot 与记忆编辑器通过 lock 文件互斥运行。

- bot 启动时检查 editor 是否在跑（edit.lock）
- editor 启动时检查 bot 是否在跑（plana.pid / systemd）
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PID_FILE = DATA_DIR / "plana.pid"
EDIT_LOCK = DATA_DIR / "edit.lock"
SERVICE = "plana-chat"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def bot_is_running() -> bool:
    """bot 是否在运行（后台 PID 或 systemd 任一）。"""
    if PID_FILE.exists():
        try:
            if _pid_alive(int(PID_FILE.read_text().strip())):
                return True
        except (ValueError, OSError):
            pass
    try:
        return subprocess.run(
            ["systemctl", "is-active", "--quiet", SERVICE],
            capture_output=True,
        ).returncode == 0
    except FileNotFoundError:
        return False


def editor_is_running() -> bool:
    """编辑器是否在运行（存在 edit.lock 且 PID 存活）。陈旧锁自动清理。"""
    if not EDIT_LOCK.exists():
        return False
    try:
        pid = int(EDIT_LOCK.read_text().strip())
    except (ValueError, OSError):
        return True  # 无法解析，保守视为占用
    if _pid_alive(pid):
        return True
    EDIT_LOCK.unlink(missing_ok=True)
    return False


def acquire_edit_lock() -> None:
    EDIT_LOCK.parent.mkdir(parents=True, exist_ok=True)
    EDIT_LOCK.write_text(str(os.getpid()))


def release_edit_lock() -> None:
    EDIT_LOCK.unlink(missing_ok=True)
