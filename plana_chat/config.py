"""配置加载：读取 config.yaml，并从 .env 注入环境变量（如 API 密钥）。"""
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

_env_loaded = False


def _ensure_dotenv() -> None:
    global _env_loaded
    if not _env_loaded:
        load_dotenv(PROJECT_ROOT / ".env")
        _env_loaded = True


class Config:
    """轻量配置对象：嵌套 dict 以属性访问，支持 get 默认值。"""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            val = self._data[name]
        except KeyError:
            raise AttributeError(name) from None
        return Config(val) if isinstance(val, dict) else val

    def get(self, name: str, default: Any = None) -> Any:
        val = self._data.get(name, default)
        return Config(val) if isinstance(val, dict) else val

    def as_dict(self) -> dict:
        return self._data


def load_config(path: Optional[str] = None) -> Config:
    _ensure_dotenv()
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(data)


def env(name: str, default: str = "") -> str:
    _ensure_dotenv()
    return os.environ.get(name, default)
