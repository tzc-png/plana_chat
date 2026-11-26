"""Telegram 机器人：httpx 长轮询，复用角色对话 + 本地语音链路。"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import signal
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import httpx

from . import lock
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
        self._greeted: dict[str, str] = {}  # {时间: 上次问候日期}，避免重复问候
        self._state_file = Path(__file__).resolve().parent.parent / "data" / "state.json"
        self._offset_file = Path(__file__).resolve().parent.parent / "data" / "offset"
        self.offset = self._load_offset()
        self.mode = self._load_mode()
        self.fifo: deque[tuple[int, str, str, str | None]] = deque()  # (chat_id, user_id, text, created_at)

    def _client(self) -> httpx.AsyncClient:
        kwargs: dict = {"timeout": 60, "trust_env": False}
        if self.proxy:
            kwargs["proxy"] = self.proxy
        return httpx.AsyncClient(**kwargs)

    async def _fetch_updates(self, timeout: int = 30) -> None:
        """拉取 updates 并按序 push 进 FIFO（确认 offset；timeout=0 用于短轮询）。"""
        async with self._client() as client:
            resp = await client.get(
                f"{self.api}/getUpdates",
                params={"offset": self.offset, "timeout": timeout},
            )
            resp.raise_for_status()
            updates = resp.json().get("result", [])
        if updates:
            self.offset = max(u["update_id"] for u in updates) + 1
            self._save_offset()
        for upd in updates:
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
            created_at = None
            date_ts = msg.get("date")
            if date_ts:
                created_at = datetime.fromtimestamp(date_ts).strftime("%Y-%m-%d %H:%M:%S")
            self.fifo.append((chat_id, user_id, text, created_at))

    async def _retry_send(self, fn) -> None:
        """发送失败按 2s/4s/… 退避重试（次数可配），最终仍失败则抛出。"""
        retries = max(1, int(self.cfg.telegram.get("send_retries", 3)))
        last_exc: Exception | None = None
        for i in range(retries):
            try:
                await fn()
                return
            except Exception as e:
                last_exc = e
                if i < retries - 1:
                    logger.warning("发送失败（%s），%d 秒后重试", type(e).__name__, (i + 1) * 2)
                    await asyncio.sleep((i + 1) * 2)
        if last_exc is not None:
            raise last_exc

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        async with self._client() as client:
            await client.post(
                f"{self.api}/sendChatAction",
                json={"chat_id": chat_id, "action": action},
            )

    async def send_text(self, chat_id: int, text: str) -> None:
        logger.info("[普拉娜] chat=%s: %s", chat_id, text)

        async def _do():
            async with self._client() as client:
                resp = await client.post(
                    f"{self.api}/sendMessage",
                    json={"chat_id": chat_id, "text": text},
                )
                resp.raise_for_status()

        await self._retry_send(_do)

    async def send_voice(self, chat_id: int, ogg: bytes) -> None:
        logger.info("[普拉娜] chat=%s: [语音]（%d 字节）", chat_id, len(ogg))

        async def _do():
            async with self._client() as client:
                resp = await client.post(
                    f"{self.api}/sendVoice",
                    data={"chat_id": str(chat_id)},
                    files={"voice": ("voice.ogg", ogg, "audio/ogg")},
                )
                resp.raise_for_status()

        await self._retry_send(_do)

    async def send_photo(self, chat_id: int, photo_path) -> None:
        path = Path(photo_path)
        logger.info("[普拉娜] chat=%s: [表情图] %s", chat_id, path.name)
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "image/png")

        async def _do():
            async with self._client() as client:
                with open(path, "rb") as f:
                    resp = await client.post(
                        f"{self.api}/sendPhoto",
                        data={"chat_id": str(chat_id)},
                        files={"photo": (path.name, f, mime)},
                    )
                resp.raise_for_status()

        await self._retry_send(_do)

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
    def _load_offset(self) -> int:
        try:
            return int(self._offset_file.read_text().strip())
        except Exception:
            return 0

    def _save_offset(self) -> None:
        try:
            self._offset_file.parent.mkdir(parents=True, exist_ok=True)
            self._offset_file.write_text(str(self.offset))
        except Exception:
            logger.warning("保存 offset 失败", exc_info=True)

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

    async def handle_text(self, chat_id: int, user_id: str, text: str, created_at: str | None = None) -> None:
        if self._handle_command(chat_id, text):
            return

        # 阶段0前：静默超时的话题自动结算（用上一轮的时间判断）
        await self.pipeline._maybe_settle_idle_topic(user_id)

        # 阶段0：老师消息立即落库 + 取消提醒
        self.pipeline.memory.cancel_reminders(user_id)
        await self.pipeline.store_user(user_id, text, created_at)

        # 阶段1：①判断（无营养静默；切话题处理旧话题）
        judge = await self.pipeline._judge_topic(user_id, text)
        if not judge["has_content"]:
            logger.info("无营养消息，静默跳过：%s", text)
            # 无营养消息也判断一次话题是否收尾，若收尾则拾起旧话题
            try:
                if await self.pipeline._topic_finished_after_no_content(user_id, text):
                    await self._settle_and_resume(chat_id, user_id, text)
            except Exception:
                logger.warning("无营养消息后的收尾判断失败", exc_info=True)
            return
        if not judge["same"]:
            cur = self.pipeline.memory.get_current_topic(user_id)
            if cur["kind"] == "emotion":
                await self.pipeline._settle_emotion(user_id, cur["topic"])
            elif cur["topic"]:
                self.pipeline._hang_topic(user_id, cur["topic"], judge.get("old_context", ""))
            self.pipeline._settle_topic_affinity(user_id)  # 旧话题好感度结算
        self.pipeline.memory.set_current_topic(user_id, judge["topic"], "normal")

        await self.send_chat_action(chat_id, "typing")

        # 阶段2+3：生成回复，发送前查 FIFO（同话题合并重生成 / 不同话题放弃）
        pending = [text]
        upgraded = None
        while True:
            combined = "（老师此前说、尚未回复）\n" + "\n".join(pending) if len(pending) > 1 else pending[0]

            # 阶段2：生成回复（纯生成 + 自检 + 申请判断 + 重生成）
            display_text, emotion = await self.pipeline.generate(user_id, combined)
            if self.cfg.behavior.get("self_review", True):
                try:
                    display_text, emotion = await self.pipeline.self_review(user_id, combined, display_text, emotion)
                except Exception:
                    logger.warning("回复自检失败，保留原文", exc_info=True)
            try:
                rel = self.pipeline.memory.get_relationship(user_id)
                emotion = await self.pipeline._emotion_review(user_id, rel["stage"], display_text, emotion)
            except Exception:
                logger.warning("情感自检失败，保留原情感", exc_info=True)
            try:
                upgraded = await self.pipeline._handle_application(user_id, combined)
                if upgraded:
                    result = await self.pipeline.regenerate_after_upgrade(user_id, combined, upgraded)
                    if result:
                        display_text, emotion = result
            except Exception:
                logger.warning("关系申请判断失败", exc_info=True)

            # 阶段3：发送前检查 FIFO（短轮询拉新消息）
            await self._fetch_updates(0)
            if not self.fifo or self.fifo[0][1] != user_id:
                break  # 无新消息，或非本用户
            nchat, nuser, ntext, ndate = self.fifo[0]  # peek 队头
            njudge = await self.pipeline._judge_topic(user_id, ntext)
            if not njudge["has_content"]:
                logger.info("忽略无营养新消息：%s", ntext)
                break  # 无营养 → 忽略（留在 FIFO），发送当前回复
            if not njudge["same"]:
                logger.info("发送前发现新话题，放弃当前回复：%s", ntext)
                return  # 不同话题 → 放弃，回主循环处理新消息
            # 同话题 → 合并
            self.fifo.popleft()
            await self.pipeline.store_user(user_id, ntext, ndate)
            pending.append(ntext)
            # 合并分支也推进 current_topic，避免停在首条消息的裸话题
            self.pipeline.memory.set_current_topic(user_id, njudge["topic"], "normal")
            logger.info("合并同话题新消息：%s", ntext)

        # 阶段4：状态修改（确认发送后）
        self.pipeline._accumulate_emotion(user_id, emotion)
        self.pipeline._handle_special_emotion(user_id, emotion, text)
        if upgraded in ("恋爱", "结婚"):
            self.pipeline._apply_upgrade(user_id, upgraded)
        self.pipeline._expire_topics(user_id)
        try:
            remind = await self.pipeline._detect_reminder(user_id, text)
            if remind:
                self.pipeline.memory.add_topic(user_id, "reminder", remind, importance=9)
                logger.info("挂起提醒任务：%s", remind)
        except Exception:
            logger.warning("提醒检测失败", exc_info=True)

        # 阶段5：语音 + 存储 + 发送
        voice_wav = await self.pipeline.make_voice(display_text)
        await self.pipeline.store_assistant(user_id, display_text, emotion)
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

        # 阶段6：续接判断
        if self.cfg.behavior.get("follow_up", True) and self.mode != "quiet":
            try:
                await asyncio.sleep(self.cfg.behavior.get("follow_up_delay", 4))
                await self._fetch_updates(0)
                if self.fifo:
                    logger.info("跳过续接：有新消息")
                else:
                    await self._topic_follow_up(chat_id, user_id, text, display_text)
            except Exception:
                logger.exception("话题续接失败")

    async def _settle_and_resume(self, chat_id: int, user_id: str, text: str) -> None:
        """结算当前话题好感度并拾起一个挂起话题（若合适）。"""
        self.pipeline._settle_topic_affinity(user_id)
        # 清空当前话题：话题已结束，无论是否拾起都不再残留（避免静默超时误挂起幽灵话题）
        self.pipeline.memory.set_current_topic(user_id, "", "normal")
        resume = await self.pipeline._resume_pending_topic(user_id)
        if resume:
            try:
                resume, _ = await self.pipeline.self_review(user_id, text, resume, "neutral")
            except Exception:
                pass
            await self._fetch_updates(0)
            if self.fifo:
                logger.info("跳过拾起：有新消息")
                return
            await self.send_text(chat_id, resume)
            await self.pipeline.store_assistant(user_id, resume)

    async def _topic_follow_up(self, chat_id: int, user_id: str, text: str, display_text: str) -> None:
        """话题续接：二态判断 finished / continuing，续接或结算+拾起。"""
        if self.pipeline.memory.has_pending_reminder(user_id):
            return
        if self.mode == "active":
            state = "continuing"
        else:
            state = await self.pipeline._topic_decision(user_id, text, display_text)

        if state == "continuing":
            cur_topic = self.pipeline.memory.get_current_topic(user_id)["topic"]
            cont = await self.pipeline._topic_continuation(user_id, cur_topic or text, text, display_text)
            if cont:
                try:
                    cont, _ = await self.pipeline.self_review(user_id, text, cont, "neutral")
                except Exception:
                    pass
                await self._fetch_updates(0)
                if self.fifo:
                    logger.info("跳过续接：有新消息")
                    return
                await self.send_text(chat_id, cont)
                await self.pipeline.store_assistant(user_id, cont)
        else:  # finished
            await self._settle_and_resume(chat_id, user_id, text)

    async def _reminder_loop(self) -> None:
        """每 60 秒检查一次是否有到期的提醒任务。"""
        while True:
            try:
                await self._check_due_reminders()
            except Exception:
                logger.exception("提醒检查出错")
            await asyncio.sleep(60)

    async def _check_due_reminders(self) -> None:
        minutes = self.cfg.greeting.get("reminder_minutes", 20)
        for chat_id in self.cfg.greeting.chat_ids:
            user_id = str(chat_id)
            for r in self.pipeline.memory.get_due_reminders(user_id, minutes=minutes):
                try:
                    text = await self.pipeline._reminder_text(user_id, r["content"])
                    if text:
                        await self.send_text(chat_id, text)
                        await self.pipeline.store_assistant(user_id, text)
                except Exception:
                    logger.exception("发送提醒失败")
                finally:
                    self.pipeline.memory.close_topic(r["id"])

    async def run(self) -> None:
        logger.info("Telegram 机器人已启动（token 前缀 %s…）", self.token[:8])
        asyncio.create_task(self._state_watcher())
        asyncio.create_task(self._reminder_loop())
        if self.cfg.greeting.enabled:
            asyncio.create_task(self._greet_loop())
            logger.info("主动问候已启用：%s", self.cfg.greeting.times)
        while True:
            try:
                # 长轮询拉取 → push FIFO
                await self._fetch_updates(30)
                # 逐条处理 FIFO
                while self.fifo:
                    chat_id, user_id, text, created_at = self.fifo.popleft()
                    logger.info("[老师] chat=%s: %s", chat_id, text)
                    try:
                        await self.handle_text(chat_id, user_id, text, created_at)
                    except Exception:
                        logger.exception("处理消息失败 chat=%s", chat_id)
                        try:
                            await self.send_text(chat_id, "老师……好像出了点问题，请稍后再试。")
                        except Exception:
                            pass
            except Exception as e:
                if isinstance(e, httpx.HTTPStatusError):
                    status = e.response.status_code
                    if status == 409:
                        # 另一个实例正在用同一 token 拉取更新，本实例重试无意义，直接退出
                        logger.warning("检测到另一个 bot 实例正在运行（409 Conflict），本实例退出")
                        break
                    logger.warning("长轮询 HTTP 错误（%s），5 秒后重试", status)
                elif isinstance(e, httpx.RequestError):
                    # 网络/请求层错误（代理断连等）只记一行，避免 traceback 刷屏
                    logger.warning("长轮询出错（%s），5 秒后重试", type(e).__name__)
                else:
                    logger.exception("长轮询出错，5 秒后重试")
                await asyncio.sleep(5)


async def main() -> None:
    # 启动前检查：记忆编辑器是否在运行（互斥）
    if lock.editor_is_running():
        print("错误：记忆编辑器正在运行，请先完成编辑再启动 bot", file=sys.stderr)
        sys.exit(1)
    # 默认 INFO；PLANA_VERBOSE=1（plana trace）时只把 plana_chat 提到 DEBUG
    verbose = os.environ.get("PLANA_VERBOSE") == "1"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # 关闭 httpx/httpcore 的日志，避免含 token 的 URL 或底层网络细节刷屏
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if verbose:
        logging.getLogger("plana_chat").setLevel(logging.DEBUG)
    # 优雅关闭：收到 SIGTERM（plana off 发的）/SIGINT（Ctrl+C）时记录日志再退出
    def _shutdown(signum, frame):
        logger.info("机器人已停止（收到信号 %s）", signum)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    bot = TelegramBot(load_config())
    await bot.run()
