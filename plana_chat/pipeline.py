"""完整链路编排：消息 → LLM → 后处理 → 翻译 → TTS → 语音。"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import Config
from .embedding import embed_text
from .llm import build_llm, build_local_llm
from .llm.base import ChatLLM
from .memory import MemoryStore
from .postprocess import clean_for_tts, extract_emotion, strip_emotion_tag
from .search import web_search
from .translator import Translator
from .tts import TTSServerManager, synthesize

logger = logging.getLogger("plana_chat.pipeline")

# 判断是否需要联网搜索的启发式关键词
_SEARCH_KEYWORDS = (
    "?", "？", "最新", "今天", "现在", "天气", "新闻", "股价", "汇率",
    "价格", "是什么", "为什么", "怎么样", "多少", "几点", "明天", "后天",
    "日期", "时间", "排名", "热搜",
)


def _should_search(text: str) -> bool:
    return any(k in text for k in _SEARCH_KEYWORDS)


@dataclass
class ProcessResult:
    display_text: str
    emotion: str
    voice_wav: bytes | None = None  # 合成语音的 WAV 字节（通道无关）


class PlanaPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.llm: ChatLLM = build_llm(cfg)
        self.local_llm: ChatLLM | None = (
            build_local_llm(cfg) if cfg.llm.get("fallback_to_local", False) else None
        )
        self.translator = Translator(cfg)
        self.memory = MemoryStore(cfg.memory.db_path)
        self.tts = TTSServerManager(cfg)

        persona_file = Path(cfg.character.persona_file)
        if not persona_file.is_absolute():
            persona_file = Path(__file__).resolve().parent.parent / persona_file
        self.persona_template = persona_file.read_text(encoding="utf-8")

    # ── prompt 构建 ──────────────────────────────────────────────────────────
    def _system_prompt(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.persona_template.replace("{current_time}", now)

    @staticmethod
    def _period(hour: int) -> str:
        if 5 <= hour < 11:
            return "早晨"
        if 11 <= hour < 14:
            return "中午"
        if 14 <= hour < 18:
            return "下午"
        if 18 <= hour < 24:
            return "晚上"
        return "深夜"

    def _hint(self, user_id: str, history: list[dict]) -> str:
        if not history:
            return "这是你和老师（用户）的第一次对话。你刚被唤醒，请表现得文静、礼貌且略带羞涩。"
        last = self.memory.last_user_time(user_id)
        if last is not None:
            diff = (datetime.now() - last).total_seconds()
            if diff > self.cfg.memory.reset_hours * 3600:
                return (
                    f"***一段时间过去了，现在是{self._period(datetime.now().hour)}，"
                    "之前的场景已结束，请根据当前时间重新开始互动。***"
                )
        return ""

    # ── 主流程 ──────────────────────────────────────────────────────────────
    async def process(self, user_id: str, user_text: str) -> ProcessResult:
        await self._maybe_consolidate(user_id)
        history = self.memory.recent(user_id, self.cfg.memory.rounds)
        user_emb = await self._embed(user_text)

        # 可选：联网搜索增强
        user_content = user_text
        if self.cfg.search.enabled and _should_search(user_text):
            try:
                results = await web_search(user_text, self.cfg)
                if results:
                    user_content = (
                        f"{user_text}\n\n"
                        "【以下是为该问题检索到的网络信息，请结合着回答，"
                        "不要照抄，也不要提及这些标注本身】\n"
                        f"{results}"
                    )
            except Exception:
                logger.warning("联网搜索失败，降级为普通回复", exc_info=True)

        messages = [{"role": "system", "content": self._system_prompt()}]

        # 注入长期记忆：摘要 + 事实 + 未完成项
        summary = self.memory.get_summary(user_id)
        if summary:
            messages.append({"role": "system", "content": f"【长期记忆摘要】\n{summary}"})
        facts = self.memory.recent_facts(user_id, "fact", limit=10)
        if facts:
            messages.append(
                {"role": "system", "content": "【关于老师的已知信息】\n" + "\n".join(f"- {f}" for f in facts)}
            )
        open_items = self.memory.recent_open_items(user_id, limit=5)
        if open_items:
            messages.append(
                {"role": "system", "content": "【老师提到但尚未完成/聊完的事（可自然带一句关心，不要生硬提醒）】\n"
                 + "\n".join(f"- {o}" for o in open_items)}
            )

        # 注入语义检索到的历史细节
        if user_emb is not None:
            retrieved = self._retrieve(user_id, user_emb)
            if retrieved:
                lines = "\n".join(
                    ("老师：" if r["role"] == "user" else "普拉娜：") + r["content"]
                    for r in retrieved
                )
                messages.append(
                    {"role": "system", "content": "【与当前话题相关的历史对话（可参考回忆细节）】\n" + lines}
                )

        messages.extend(history)
        hint = self._hint(user_id, history)
        if hint:
            messages.append({"role": "system", "content": hint})
        messages.append({"role": "user", "content": user_content})

        raw = await self._chat_with_fallback(messages)

        emotion = extract_emotion(raw)
        display_text = strip_emotion_tag(raw)

        # 存储（带 embedding，原文永久保留）
        self.memory.add(user_id, "user", user_text, embedding=user_emb)
        assistant_emb = await self._embed(display_text)
        self.memory.add(user_id, "assistant", display_text, emotion, embedding=assistant_emb)

        voice_wav = None
        mode = self.cfg.reply.mode
        if mode in ("text_voice", "voice"):
            voice_wav = await self._voice(display_text)
        return ProcessResult(display_text=display_text, emotion=emotion, voice_wav=voice_wav)

    async def _chat_with_fallback(self, messages: list[dict]) -> str:
        try:
            return await self.llm.chat(messages)
        except Exception as e:
            if self.local_llm is not None and self.local_llm is not self.llm:
                reason = self._degrade_reason(str(e))
                if reason:
                    logger.warning(
                        "\033[31m[降级] 云端不可用（%s），已自动切换到本地 Ollama\033[0m", reason
                    )
                else:
                    logger.warning("[降级] 云端调用失败，已切换到本地 Ollama：%s", e)
                return await self.local_llm.chat(messages)
            raise

    @staticmethod
    def _degrade_reason(err: str) -> str | None:
        """判断云端失败是否属于余额/额度类（用于红色高亮提示）。"""
        low = err.lower()
        for kw in ("insufficient", "balance", "余额", "402", "quota", "exceeded", "rate limit"):
            if kw in low:
                return err
        return None

    # ── 记忆整理（总结压缩 + 事实抽取）──────────────────────────────────────
    @staticmethod
    def _format_msgs(msgs: list[dict]) -> str:
        lines = []
        for m in msgs:
            who = "老师" if m["role"] == "user" else "普拉娜"
            lines.append(f"{who}: {m['content']}")
        return "\n".join(lines)

    async def _summarize(self, old_msgs: list[dict], existing: str | None) -> str:
        prompt = (
            "请把下面这段老师和普拉娜的对话历史，压缩成一段简洁的「长期记忆摘要」，"
            "保留关键信息（重要事件、老师提到的个人信息、情感关系进展），控制在 200 字以内。"
            "如果已有摘要，请融合新旧内容（新的信息在前，旧的简略带过）。\n\n"
            f"【已有摘要】\n{existing or '（无）'}\n\n"
            f"【新增对话】\n{self._format_msgs(old_msgs)}\n\n"
            "只输出新的摘要文本。"
        )
        raw = await self._chat_with_fallback([{"role": "user", "content": prompt}])
        return raw.strip()

    async def _extract_facts(self, old_msgs: list[dict]) -> tuple[list[str], list[str]]:
        prompt = (
            "请从下面这段老师和普拉娜的对话中，抽取两类信息：\n"
            "1. 「关于老师的事实」：姓名、喜好、习惯、重要个人信息等稳定事实\n"
            "2. 「老师提到但尚未完成/未解决的事项」：待办、计划、进行中的话题等\n\n"
            f"对话：\n{self._format_msgs(old_msgs)}\n\n"
            '请严格只输出 JSON，格式：{"facts": ["..."], "open_items": ["..."]}，没有的类别输出空数组 []。'
        )
        raw = await self._chat_with_fallback([{"role": "user", "content": prompt}])
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return [], []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return [], []
        return data.get("facts", []), data.get("open_items", [])

    async def _maybe_consolidate(self, user_id: str) -> None:
        """历史消息超阈值时：总结压缩 + 事实抽取（原文保留，不删除）。"""
        threshold = self.cfg.memory.get("compact_threshold", 20)
        upto = self.memory.consolidated_upto(user_id)
        batch = self.memory.unconsolidated(user_id, upto, limit=200)
        if len(batch) < threshold:
            return
        try:
            new_summary = await self._summarize(batch, self.memory.get_summary(user_id))
            self.memory.update_summary(user_id, new_summary)
            facts, open_items = await self._extract_facts(batch)
            self.memory.add_facts(user_id, facts, "fact")
            self.memory.add_facts(user_id, open_items, "open_item")
            self.memory.update_consolidated_upto(user_id, batch[-1]["id"])
            logger.info(
                "记忆整理完成 user=%s（处理 %d 条，事实 %d、未完成项 %d）",
                user_id, len(batch), len(facts), len(open_items),
            )
        except Exception:
            logger.warning("记忆整理失败 user=%s", user_id, exc_info=True)

    async def _embed(self, text: str) -> bytes | None:
        """本地 embedding，返回 float32 字节；失败返回 None。"""
        try:
            emb = await embed_text(
                text,
                host=self.cfg.ollama.host,
                model=self.cfg.memory.get("embedding_model", "bge-m3"),
            )
            return np.asarray(emb, dtype=np.float32).tobytes()
        except Exception:
            logger.warning("embedding 失败", exc_info=True)
            return None

    def _retrieve(self, user_id: str, user_emb: bytes) -> list[dict]:
        """语义检索与当前话题最相关的历史消息。"""
        top_k = self.cfg.memory.get("retrieve_top_k", 5)
        exclude = self.cfg.memory.rounds * 2
        return self.memory.search_similar(user_id, user_emb, limit=top_k, exclude_recent=exclude)

    async def greet(self, user_id: str) -> ProcessResult | None:
        """上下文感知的主动问候；最近有进行中对话时返回 None（跳过）。"""
        now = datetime.now()

        # 最近 30 分钟内有对话（进行中）→ 跳过定时问候
        last_time = self.memory.last_message_time(user_id)
        if last_time is not None and (now - last_time).total_seconds() < 30 * 60:
            logger.info("跳过主动问候：最近有进行中对话 user=%s", user_id)
            return None

        hour = now.hour
        period = "早上" if 5 <= hour < 12 else ("晚上" if hour >= 18 else "现在")
        recent = self.memory.recent(user_id, self.cfg.memory.rounds)
        background = self._format_msgs(recent[-6:]) if recent else "（暂无历史，这是第一次问候）"

        hint = (
            f"现在是 {now.strftime('%H:%M')}（{period}）。请主动给老师发一句问候，"
            "并参考下面的「最近对话背景」判断情境、做出合理反应：\n"
            "- 先仔细分辨老师最近的原话：\n"
            "  * 若明确说「睡了/晚安/睡觉/要睡了」等，才可演绎成「偷偷开门看了一眼，发现老师已经睡着了」之类的轻声自言自语\n"
            "  * 若只说「休息/先休息/去休息」等，可能只是回去歇着、不一定睡着，就正常关心一句（如「好好休息」），不要演绎成睡着了\n"
            "- 若老师提到生病、不舒服、熬夜、加班等，要表现出关心\n"
            "- 若没有特殊情况，就按当前时间自然问候\n"
            "问候要简短自然，动作可写在括号里，末尾加情感标签 [emotion:xxx]。\n\n"
            f"【最近对话背景】\n{background}\n\n"
            "只输出问候内容。"
        )

        open_items = self.memory.recent_open_items(user_id, limit=3)
        if open_items:
            hint += "\n\n另外老师之前提到这些还没完成的事，可自然带一句关心（不要生硬）：" + "；".join(open_items)

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": hint},
        ]
        raw = await self._chat_with_fallback(messages)

        emotion = extract_emotion(raw)
        display_text = strip_emotion_tag(raw)
        voice_wav = None
        if self.cfg.reply.mode in ("text_voice", "voice"):
            voice_wav = await self._voice(display_text)
        return ProcessResult(display_text=display_text, emotion=emotion, voice_wav=voice_wav)

    async def follow_up(self, user_id: str, last_user: str, last_assistant: str, dynamic: bool = False) -> str | None:
        """生成一句简短自然的追问；dynamic=True 时由 LLM 动态判断是否需要追问。"""
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": last_user},
            {"role": "assistant", "content": last_assistant},
        ]
        if dynamic:
            messages.append({"role": "user", "content": (
                "根据上面的对话，判断你是否需要再追问/关心一句：\n"
                "- 若对话已自然收尾、或老师没有继续的意思，则不需要\n"
                "- 若还想继续关心或推进话题，则输出一句简短自然的追问（15 字以内）\n"
                '严格只输出 JSON：{"should": true, "text": "追问内容"} 或 {"should": false}'
            )})
            raw = await self._chat_with_fallback(messages)
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                return None
            try:
                data = json.loads(m.group(0))
            except Exception:
                return None
            if not data.get("should"):
                return None
            text = str(data.get("text", "")).strip()
            return text or None
        else:
            messages.append({"role": "user", "content": (
                "请再补一句简短、自然的追问或关心（一句话，尽量 15 字以内），"
                "让对话继续。不要啰嗦、不要重复刚才的话、不要带情感标签。"
            )})
            raw = await self._chat_with_fallback(messages)
            text = strip_emotion_tag(raw).strip()
            return text or None

    async def store_assistant(self, user_id: str, text: str, emotion: str | None = None) -> None:
        """把追问等补充回复也存入记忆（带 embedding）。"""
        emb = await self._embed(text)
        self.memory.add(user_id, "assistant", text, emotion, embedding=emb)

    async def _voice(self, display_text: str) -> bytes | None:
        try:
            voice_text = clean_for_tts(display_text)
            max_chars = self.cfg.reply.get("max_voice_chars", 0)
            if max_chars and len(voice_text) > max_chars:
                voice_text = voice_text[:max_chars]
            if not voice_text:
                return None

            ja = await self.translator.translate(voice_text)
            if not ja:
                return None

            # TTS 为阻塞操作，放到线程避免卡事件循环
            return await asyncio.to_thread(self._synth_blocking, ja)
        except Exception:
            # 语音失败不影响文字回复
            return None

    def _synth_blocking(self, ja: str) -> bytes:
        self.tts.ensure_running()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            synthesize(ja, wav_path, self.cfg.tts.socket, self.cfg.tts.lang)
            return Path(wav_path).read_bytes()
        finally:
            Path(wav_path).unlink(missing_ok=True)


def build_pipeline(cfg: Config) -> PlanaPipeline:
    return PlanaPipeline(cfg)
