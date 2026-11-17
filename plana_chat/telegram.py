"""Telegram 机器人：httpx 长轮询，复用角色对话 + 本地语音链路。"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
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
        logger.info("[普拉娜] chat=%s: %s", chat_id, text)
        async with self._client() as client:
            resp = await client.post(
                f"{self.api}/sendMessage",
                json={"chat_id": chat_id, "text": text},
            )
            resp.raise_for_status()

    async def send_voice(self, chat_id: int, ogg: bytes) -> None:
        logger.info("[普拉娜] chat=%s: [语音]（%d 字节）", chat_id, len(ogg))
        async with self._client() as client:
            resp = await client.post(
                f"{self.api}/sendVoice",
                data={"chat_id": str(chat_id)},
                files={"voice": ("voice.ogg", ogg, "audio/ogg")},
            )
            resp.raise_for_status()

    async def send_photo(self, chat_id: int, photo_path) -> None:
        path = Path(photo_path)
        logger.info("[普拉娜] chat=%s: [表情图] %s", chat_id, path.name)
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

    async def _drain_recent_user_messages(self, user_id: str, within_seconds: int = 15) -> list[str]:
        """发送前检查：消费该用户最近 N 秒内新发来的消息（用于合并重答）。"""
        now = time.time()
        async with self._client() as client:
            resp = await client.get(
                f"{self.api}/getUpdates",
                params={"offset": self.offset, "timeout": 0},
            )
            resp.raise_for_status()
            updates = resp.json().get("result", [])
        texts = []
        for upd in updates:
            self.offset = max(self.offset, upd["update_id"] + 1)
            msg = upd.get("message")
            if not msg or "text" not in msg:
                continue
            muser = str(msg.get("from", {}).get("id", msg.get("chat", {}).get("id")))
            if muser != user_id or now - msg.get("date", 0) > within_seconds:
                continue
            texts.append(msg["text"].strip())
        return texts

    async def _peek_recent_user_messages(self, user_id: str, within_seconds: int = 15) -> bool:
        """检查（不消费）该用户最近 N 秒内是否有新消息。"""
        now = time.time()
        async with self._client() as client:
            resp = await client.get(
                f"{self.api}/getUpdates",
                params={"offset": self.offset, "timeout": 0},
            )
            resp.raise_for_status()
            updates = resp.json().get("result", [])
        for upd in updates:
            msg = upd.get("message")
            if not msg or "text" not in msg:
                continue
            muser = str(msg.get("from", {}).get("id", msg.get("chat", {}).get("id")))
            if muser == user_id and now - msg.get("date", 0) <= within_seconds:
                return True
        return False

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

        # 1. 生成回复（暂不存储）
        display_text, emotion = await self.pipeline.generate(user_id, text)

        # 2. 发送前短暂等待，让用户可能正在输入的消息到达
        settle = self.cfg.behavior.get("settle_delay", 1.5)
        if settle:
            await asyncio.sleep(settle)

        # 3. 发送前检查是否有新消息（用户连续发消息时合并重答）
        new_texts = await self._drain_recent_user_messages(user_id)
        if new_texts:
            combined = text + "\n" + "\n".join(new_texts)
            logger.info("[老师] 合并新消息：%s", " | ".join(new_texts))
            display_text, emotion = await self.pipeline.generate(user_id, combined)
            text = combined

        # 4. 回复自检（检查矛盾/不合时宜，有问题则修正）
        if self.cfg.behavior.get("self_review", True):
            try:
                display_text, emotion = await self.pipeline.self_review(
                    user_id, text, display_text, emotion
                )
            except Exception:
                logger.warning("回复自检失败，保留原文", exc_info=True)

        # 5. 生成语音 + 存储
        voice_wav = await self.pipeline.make_voice(display_text)
        await self.pipeline.store(user_id, text, display_text, emotion)

        # 6. 发送
        mode = self.cfg.reply.mode
        if mode in ("text_voice", "text"):
            await self.send_text(chat_id, display_text)
        if mode in ("text_voice", "voice") and voice_wav:
            try:
                ogg = await asyncio.to_thread(wav_to_ogg_opus, voice_wav)
                await self.send_voice(chat_id, ogg)
            except Exception:
                logger.exception("发送语音失败")
        await self._maybe_send_emotion(chat_id, emotion)

        # 追问：active 总是追问，default 动态决策，quiet 不追问
        if self.cfg.behavior.get("follow_up", True) and self.mode != "quiet":
            try:
                await asyncio.sleep(self.cfg.behavior.get("follow_up_delay", 4))
                # 追问前检查是否有新消息，有则跳过（对话已推进，避免追问过时）
                if await self._peek_recent_user_messages(user_id):
                    logger.info("跳过追问：等待期间有新消息")
                else:
                    follow = await self.pipeline.follow_up(
                        user_id, text, display_text, dynamic=(self.mode == "default")
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
                # 一次性确认整个批次，避免发送前 drain 重复拉取
                if updates:
                    self.offset = max(u["update_id"] for u in updates) + 1
                i = 0
                n = len(updates)
                while i < n:
                    upd = updates[i]
                    msg = upd.get("message")
                    i += 1
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
                    # 合并同一用户短时间内（60 秒内）的连续消息，一并回复
                    parts = [text]
                    ts = msg.get("date", 0)
                    while i < n:
                        nmsg = updates[i].get("message")
                        if not (nmsg and "text" in nmsg):
                            break
                        nuser = str(nmsg.get("from", {}).get("id", nmsg.get("chat", {}).get("id")))
                        if nuser == user_id and 0 <= abs(nmsg.get("date", 0) - ts) <= 60:
                            parts.append(nmsg["text"].strip())
                            i += 1
                        else:
                            break
                    combined = "\n".join(p for p in parts if p)
                    logger.info("[老师] chat=%s: %s", chat_id, combined)
                    try:
                        await self.handle_text(chat_id, user_id, combined)
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
