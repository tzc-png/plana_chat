"""回复后处理：情感标签提取与文本清洗（移植自 shittim 逻辑）。"""
from __future__ import annotations

import re

# 情感白名单（与 shittim / GUI 立绘表情一致）
EMOTION_WHITELIST = {
    "neutral",
    "sheepish",
    "love",
    "sulky",
    "jealous",
    "expectant",
    "dumbfounded",
    "delighted",
    "cheerful",
    "puzzled",
    "like",
}

# 兼容 [emotion:xxx] 与全角 （emotion:xxx）/半角 (emotion:xxx) 括号
_EMOTION_RE = re.compile(r"[\[（(]emotion\s*:\s*([^\]）)]*)[\]）)]")
_BRACKET_ACTION_RE = re.compile(r"\([^()]*\)|（[^（）]*）")
_DASH_RE = re.compile(r"[-—].*", re.S)


def extract_emotion(text: str) -> str:
    """提取 [emotion:xxx]，白名单校验，失败回退 neutral。"""
    m = _EMOTION_RE.search(text)
    if not m:
        return "neutral"
    emo = m.group(1).strip()
    return emo if emo in EMOTION_WHITELIST else "neutral"


def strip_emotion_tag(text: str) -> str:
    """彻底擦除 [emotion:xxx] 标签（用于显示与后续清洗）。"""
    return _EMOTION_RE.sub("", text).strip()


def clean_for_tts(text: str) -> str:
    """提取用于语音合成的纯台词：去括号动作 + 去破折号引导，返回说话内容。"""
    t = _BRACKET_ACTION_RE.sub("", text)
    t = _DASH_RE.sub("", t)
    return t.strip()
