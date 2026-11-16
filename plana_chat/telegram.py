"""Telegram 机器人：httpx 长轮询，复用角色对话 + 本地语音链路。"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import httpx

from .audio import wav_to_ogg_opus
from .config import Config, env, load_config
from .pipeline import build_pipeline

logger = logging.getLogger("plana_chat.telegram")


class TelegramBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.token = env(cfg.telegram.token_env)
        if not self.token:
            raise RuntimeError("缺少 Telegram Bot Token，请在 .env 里设置 TELEGRAM_BOT_TOKEN")
        self.api = f"https://api.telegram.org/bot{self.token}"
        self.proxy = cfg.telegram.get("proxy", "")
        self.pipeline = build_pipeline(cfg)
        self.offset = 0
        self._greeted: dict[str, str] = {}  # {时间: 上次问候日期}，避免重复问候
        self._state_file = Path(__file__).resolve().parent.parent / "data" / "state.json"
        self.mode = self._load_mode()

    def _client(self) -> httpx.AsyncClient:
        kwargs: dict = {"timeout": 60, "trust_env": False}
        if self.proxy:
            kwargs["proxy"] = self.proxy
        return httpx.AsyncClient(**kwargs)

    async def get_updates(self) -> list[dict]:
        async with self._client() as client:
            resp = await client.get(
                f"{self.api}/getUpdates",
                params={"offset": self.offset, "timeout": 30},
            )
            resp.raise_for_status()
            return resp.json().get("result", [])

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        async with self._client() as client:
            await client.post(
                f"{self.api}/sendChatAction",
                json={"chat_id": chat_id, "action": action},
            )

    async def send_text(self, chat_id: int, text: str) -> None:
        async with self._client() as client:
            resp = await client.post(
                f"{self.api}/sendMessage",
                json={"chat_id": chat_id, "text": text},
            )
            resp.raise_for_status()

    async def send_voice(self, chat_id: int, ogg: bytes) -> None:
        async with self._client() as client:
            resp = await client.post(
                f"{self.api}/sendVoice",
                data={"chat_id": str(chat_id)},
                files={"voice": ("voice.ogg", ogg, "audio/ogg")},
            )
            resp.raise_for_status()

    async def send_photo(self, chat_id: int, photo_path) -> None:
        path = Path(photo_path)
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "image/png")
        async with self._client() as client:
            with open(path, "rb") as f:
                resp = await client.post(
                    f"{self.api}/sendPhoto",
                    data={"chat_id": str(chat_id)},
                    files={"photo": (path.name, f, mime)},
                )
            resp.raise_for_status()

    def _emotion_image(self, emotion: str):
        """从 emotion 对应的文件夹里随机选一张表情图；无图返回 None。"""
        img_dir = Path(self.cfg.emotion.image_dir)
        if not img_dir.is_absolute():
            img_dir = Path(__file__).resolve().parent.parent / img_dir
        folder = img_dir / emotion
        if not folder.is_dir():
            return None
        imgs = [p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
        return random.choice(imgs) if imgs else None

    async def _maybe_send_emotion(self, chat_id: int, emotion: str) -> None:
        if not self.cfg.emotion.get("enabled", False):
            return
        img = self._emotion_image(emotion)
        if img is None:
            return
        try:
            await self.send_photo(chat_id, img)
        except Exception:
            logger.exception("发送表情图失败 emotion=%s", emotion)

    # ── 模式状态（quiet / active / default）──────────────────────────────
    def _load_mode(self) -> str:
        try:
            data = json.loads(self._state_file.read_text())
            return data.get("mode", "default")
        except Exception:
            return self.cfg.behavior.get("mode", "default")

    def _set_mode(self, val: str) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(json.dumps({"mode": val}))
        self.mode = val

    def _handle_command(self, chat_id: int, text: str) -> bool:
        """处理控制指令；返回 True 表示已处理（不回复）。"""
        t = text.strip().lower()
        if t in ("/quiet", "/active", "/default"):
            self._set_mode(t[1:])
            logger.info("切换模式为 %s（chat=%s）", t[1:], chat_id)
            return True
        return False

    def _is_allowed(self, user_id: str) -> bool:
        """白名单检查；列表为空则不限制。"""
        allow = self.cfg.telegram.get("allow_user_ids", [])
        if not allow:
            return True
        return user_id in {str(u) for u in allow}

    async def _state_watcher(self) -> None:
        """每 30 秒刷新模式状态，让 plana quiet/active/default（CLI）也能即时生效。"""
        while True:
            await asyncio.sleep(30)
            try:
                self.mode = self._load_mode()
            except Exception:
                pass

    async def _send_greeting(self) -> None:
        for chat_id in self.cfg.greeting.chat_ids:
            try:
                result = await self.pipeline.greet(str(chat_id))
                if result is None:
                    logger.info("跳过问候：最近有进行中对话 chat=%s", chat_id)
                    continue
                mode = self.cfg.reply.mode
                if mode in ("text_voice", "text"):
                    await self.send_text(chat_id, result.display_text)
                if mode in ("text_voice", "voice") and result.voice_wav:
                    ogg = await asyncio.to_thread(wav_to_ogg_opus, result.voice_wav)
                    await self.send_voice(chat_id, ogg)
                await self._maybe_send_emotion(chat_id, result.emotion)
            except Exception:
                logger.exception("发送问候失败 chat=%s", chat_id)

    async def _greet_loop(self) -> None:
        """每 30 秒检查一次是否到了问候时刻（目标时刻起 5 分钟窗口内，当天只发一次）。"""
        while True:
            try:
                if self.mode == "quiet":
                    await asyncio.sleep(60)
                    continue
                now = datetime.now()
                today = now.strftime("%Y-%m-%d")
                now_min = now.hour * 60 + now.minute
                for t in self.cfg.greeting.times:
                    hh, mm = (int(x) for x in t.split(":"))
                    target_min = hh * 60 + mm
                    if 0 <= now_min - target_min < 5 and self._greeted.get(t) != today:
                        self._greeted[t] = today
                        logger.info("触发主动问候 %s", t)
                        await self._send_greeting()
            except Exception:
                logger.exception("问候调度出错")
            await asyncio.sleep(30)

    async def handle_text(self, chat_id: int, user_id: str, text: str) -> None:
        if self._handle_command(chat_id, text):
            return

        await self.send_chat_action(chat_id, "typing")
        result = await self.pipeline.process(user_id, text)

        mode = self.cfg.reply.mode
        if mode in ("text_voice", "text"):
            await self.send_text(chat_id, result.display_text)
        if mode in ("text_voice", "voice") and result.voice_wav:
            try:
                ogg = await asyncio.to_thread(wav_to_ogg_opus, result.voice_wav)
                await self.send_voice(chat_id, ogg)
            except Exception:
                logger.exception("发送语音失败")
        await self._maybe_send_emotion(chat_id, result.emotion)

        # 追问：active 总是追问，default 动态决策，quiet 不追问
        if self.cfg.behavior.get("follow_up", True) and self.mode != "quiet":
            try:
                await asyncio.sleep(self.cfg.behavior.get("follow_up_delay", 4))
                follow = await self.pipeline.follow_up(
                    user_id, text, result.display_text, dynamic=(self.mode == "default")
                )
                if follow:
                    await self.send_text(chat_id, follow)
                    await self.pipeline.store_assistant(user_id, follow)
            except Exception:
                logger.exception("追问失败")

    async def run(self) -> None:
        logger.info("Telegram 机器人已启动（token 前缀 %s…）", self.token[:8])
        asyncio.create_task(self._state_watcher())
        if self.cfg.greeting.enabled:
            asyncio.create_task(self._greet_loop())
            logger.info("主动问候已启用：%s", self.cfg.greeting.times)
        while True:
            try:
                updates = await self.get_updates()
                for upd in updates:
                    self.offset = upd["update_id"] + 1
                    msg = upd.get("message")
                    if not msg or "text" not in msg:
                        continue
                    chat_id = msg["chat"]["id"]
                    user_id = str(msg.get("from", {}).get("id", chat_id))
                    text = msg["text"].strip()
                    if not text:
                        continue
                    if not self._is_allowed(user_id):
                        logger.warning(
                            "\033[31m[拦截] 非白名单用户访问已忽略（user=%s）\033[0m", user_id
                        )
                        continue
                    logger.info("[tg] chat=%s: %s", chat_id, text)
                    try:
                        await self.handle_text(chat_id, user_id, text)
                    except Exception:
                        logger.exception("处理消息失败 chat=%s", chat_id)
                        try:
                            await self.send_text(chat_id, "老师……好像出了点问题，请稍后再试。")
                        except Exception:
                            pass
            except Exception:
                logger.exception("长轮询出错，5 秒后重试")
                await asyncio.sleep(5)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # 关闭 httpx 的 INFO 日志，避免把含 token 的完整 URL 打进日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    bot = TelegramBot(load_config())
    await bot.run()
