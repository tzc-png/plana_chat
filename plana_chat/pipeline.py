"""完整链路编排：消息 → LLM → 后处理 → 翻译 → TTS → 语音。"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
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


# 好感度加减分（普拉娜输出的情感标签 → 分数）
EMOTION_SCORES = {
    "neutral": 0,
    "like": 2,
    "love": 3,
    "sheepish": 1,
    "delighted": 2,
    "cheerful": 1,
    "expectant": 1,
    "puzzled": 0,
    "dumbfounded": -1,
    "sulky": 0,
    "jealous": -1,
}

# 每轮对话的基础好感度（与情感分叠加，同样受 affinity_scale 调控）。
# 陌生阶段强制升温、熟悉阶段缓升，避免关系阶段因情感标签长期 neutral 而停滞。
STAGE_BASE_SCORES = {"陌生": 2, "熟悉": 1}

# 每阶段的关系指令（追加在基础人格之后，含阶段专属动作、彩蛋、吃醋梯度）
STAGE_PROMPTS = {
    "陌生": "【当前关系】你们刚认识。保持礼貌克制，带一点疏离感，会偷偷观察老师。动作克制，不主动肢体接触，也不说太亲昵的话。老师提到别的女生时，反应很淡。",
    "熟悉": "【当前关系】你们已经熟悉了。可以放松自然，偶尔主动关心，有一些小习惯。动作可以轻微自然（递东西、并肩走），但保持分寸。",
    "喜欢": "【当前关系】你对老师有好感，会不自觉地多在意、多关心，偶尔害羞、口是心非。动作可以有轻微的身体靠近，但还不敢太亲昵。你刚意识到这份好感，第一次说出「喜欢」时自己也会吓一跳。老师提到别的女生时会有点在意。",
    "恋爱": "【当前关系】你们是恋人（热恋期）。亲昵、依赖、撒娇都自然，会想黏着老师；动作可以牵手、挽手臂、靠肩、拥抱。刚成为恋人时，第一次牵手会害羞。保留普拉娜克制的底色。老师提到别的女生时，吃醋反应会比之前更明显。",
    "结婚": "【当前关系】你们已经结婚。稳定默契、安心自然，偶尔流露老夫老妻的平淡幸福。动作自然亲密，不再羞涩。老师提到别的女生，会露出轻微的吃醋，但更多是信任。",
}

# 阶段阈值（好感度达到才可进入该阶段；恋爱/结婚为申请阈值）
STAGE_THRESHOLDS = {"陌生": 0, "熟悉": 11, "喜欢": 31, "恋爱": 60, "结婚": 100}

# LLM 调用方函数名 → 中文标签（用于 DEBUG 日志标注是哪一步）
_CALLER_LABELS = {
    "generate": "主回复",
    "self_review": "自检",
    "_emotion_review": "情感自检",
    "_check_application": "申请判断",
    "_topic_decision": "续接判断",
    "_topic_continuation": "续接生成",
    "_judge_topic": "话题判断",
    "_can_chat": "是否方便",
    "_topic_suitability": "合适性判断",
    "_resume_topic_text": "拾起话题",
    "_resume_emotion_text": "拾起吃醋",
    "_judge_soothing": "安抚判断",
    "_detect_reminder": "提醒判断",
    "_reminder_text": "提醒文案",
    "greet": "问候",
    "_summarize": "记忆摘要",
    "_extract_facts": "事实抽取",
}

# 关系升级/申请时的「特殊时刻」语气（用于申请命中后重新生成回复）
SPECIAL_MOMENTS = {
    "恋爱": "【特殊时刻】就在刚才，老师向你表白了。请明确、真诚地接受这份心意——答应和老师在一起。回复要明确带出「接受」的语义（如「我愿意」「嗯，我们在一起吧」），不要含糊、不要顾左右而言他。同时表达出惊喜、害羞又幸福的心情。",
    "结婚": "【特殊时刻】就在刚才，老师向你求婚/你们决定结婚。请明确、郑重地接受——愿意和老师结婚。回复要明确带出「接受」的语义（如「我愿意」「好，我们结婚吧」），不要含糊。同时表达出感动、郑重、眼眶微热的心情。",
    "reject": "【特殊时刻】老师刚才向你表白/求婚了，但你还没准备好、好感度还不够。请给一个委婉、温柔的拒绝/犹豫回应（如「老师突然说这个……我还没准备好」），既不要冷漠拒绝，也不要含糊到让老师误会。",
}


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
    def _system_prompt(self, user_id: str, special_hint: str | None = None) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        base = self.persona_template.replace("{current_time}", now)
        rel = self.memory.get_relationship(user_id)
        stage_instruction = STAGE_PROMPTS.get(rel["stage"], "")
        if stage_instruction:
            base = base + "\n\n" + stage_instruction
        if special_hint:
            base = base + "\n\n" + special_hint
        return base

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
    async def generate(self, user_id: str, user_text: str, special_hint: str | None = None) -> tuple[str, str]:
        """生成回复（不存储）：返回 (display_text, emotion)。"""
        await self._maybe_consolidate(user_id)
        history = self._drop_trailing_user(self.memory.recent(user_id, self.cfg.memory.rounds))
        user_emb = await self._embed(user_text)
        if user_emb is not None:
            self.memory.update_latest_user_embedding(user_id, user_emb)

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

        messages = [{"role": "system", "content": self._system_prompt(user_id, special_hint)}]

        # 注入长期记忆：摘要 + 事实 + 未完成项
        summary, summary_at = self.memory.get_summary_with_time(user_id)
        if summary:
            header = "【长期记忆摘要】"
            if summary_at:
                header += f"（最近整理于 {summary_at}）"
            messages.append({"role": "system", "content": f"{header}\n{summary}"})
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
        return display_text, emotion

    async def store_user(self, user_id: str, user_text: str) -> None:
        """阶段0：老师消息立即落库（只存文本，embedding 延迟到阶段2 计算回填）。"""
        self.memory.add(user_id, "user", user_text, embedding=None)

    async def make_voice(self, display_text: str) -> bytes | None:
        """生成语音（WAV 字节）；纯文字模式返回 None。"""
        if self.cfg.reply.mode in ("text_voice", "voice"):
            return await self._voice(display_text)
        return None

    async def self_review(self, user_id: str, user_text: str, display_text: str, emotion: str) -> tuple[str, str]:
        """回复自检：检查与上下文是否矛盾/不合时宜，有问题则修正。"""
        recent = self._drop_trailing_user(self.memory.recent(user_id, 3))
        context = self._format_msgs(recent) if recent else "（无）"
        prompt = (
            "请审查下面这条即将发送给老师的回复。只有当你发现它存在以下问题时才修改，否则原样输出：\n"
            "1. 与前面对话上下文矛盾、前后不一致\n"
            "2. 不合时宜（老师说要工作/休息/安静，回复却在打扰或发起互动）\n"
            "3. 不符合普拉娜克制、安静、情感陪伴的性格\n"
            "4. 与普拉娜刚说过的上一句语义重复、冗余，或时态错乱（老师正在做/刚说要做的事，却用完成时去问）\n\n"
            f"【前面的对话】\n{context}\n\n"
        )
        if user_text:
            prompt += f"【老师刚说】{user_text}\n\n"
        prompt += (
            f"【待审查的回复】\n{display_text}\n\n"
            "只输出最终要发送的回复（保持情感标签 [emotion:xxx] 格式）。"
        )
        messages = [
            {"role": "system", "content": self._system_prompt(user_id)},
            {"role": "user", "content": prompt},
        ]
        raw = await self._chat_with_fallback(messages)
        new_emotion = extract_emotion(raw)
        new_text = strip_emotion_tag(raw)
        if not new_text:
            return display_text, emotion
        return new_text, new_emotion

    async def _chat_with_fallback(self, messages: list[dict]) -> str:
        try:
            raw = await self.llm.chat(messages)
        except Exception as e:
            if self.local_llm is not None and self.local_llm is not self.llm:
                reason = self._degrade_reason(str(e))
                if reason:
                    logger.warning(
                        "\033[31m[降级] 云端不可用（%s），已自动切换到本地 Ollama\033[0m", reason
                    )
                else:
                    logger.warning("[降级] 云端调用失败，已切换到本地 Ollama：%s", e)
                raw = await self.local_llm.chat(messages)
            else:
                raise
        last_user = messages[-1]["content"] if messages else ""
        # 从调用栈推断当前是哪一步（主回复/自检/申请判断等）
        frame = inspect.currentframe()
        caller = frame.f_back.f_code.co_name if frame and frame.f_back else ""
        label = _CALLER_LABELS.get(caller, caller)
        tag = f"[{label}]" if label else ""
        req = last_user[:80].replace("\n", " ")
        req += "…" if len(last_user) > 80 else ""
        logger.debug("LLM 请求%s：%s", tag, req)
        ret = raw[:80].replace("\n", " ")
        ret += "…" if len(raw) > 80 else ""
        logger.debug("LLM 返回%s：%s", tag, ret)
        return raw

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

    @staticmethod
    def _drop_trailing_user(history: list[dict]) -> list[dict]:
        """去掉历史末尾连续的老师消息——它们是本轮刚写入、会再单独作为当前消息传入的内容。"""
        out = list(history)
        while out and out[-1].get("role") == "user":
            out.pop()
        return out

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
        logger.debug("embedding 请求：%s", text[:40].replace("\n", " "))
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

    # ── 关系系统（好感度 + 阶段 + 申请）─────────────────────────────────
    def _compute_stage(self, affinity: int, current_stage: str) -> str:
        if current_stage == "结婚":
            return "结婚"  # 锁定，不降
        if current_stage == "恋爱" and affinity < STAGE_THRESHOLDS["恋爱"]:
            return "喜欢"
        if current_stage == "喜欢" and affinity < STAGE_THRESHOLDS["喜欢"]:
            return "熟悉"
        if current_stage == "熟悉" and affinity < STAGE_THRESHOLDS["熟悉"]:
            return "陌生"
        if current_stage == "陌生" and affinity >= STAGE_THRESHOLDS["熟悉"]:
            return "熟悉"
        if current_stage == "熟悉" and affinity >= STAGE_THRESHOLDS["喜欢"]:
            return "喜欢"
        return current_stage

    # ── 话题级好感度 ────────────────────────────────────────────────────
    def _accumulate_emotion(self, user_id: str, emotion: str) -> None:
        """当前话题累积一个情感标签（确认发送后调用，立即持久化）。"""
        self.memory.accumulate_topic_emotion(user_id, emotion)
        logger.debug("累积情感：%s", emotion)

    def _clear_topic_emotions(self, user_id: str) -> None:
        self.memory.clear_topic_emotions(user_id)

    def _settle_topic_affinity(self, user_id: str) -> dict:
        """结算当前话题的好感度（话题 finished 或切走时触发）。"""
        emotions = self.memory.get_current_topic(user_id)["emotions"]
        if not emotions:
            return self.memory.get_relationship(user_id)

        avg = sum(EMOTION_SCORES.get(e, 0) for e in emotions) / len(emotions)
        emotion_score = int(math.ceil(abs(avg))) * (1 if avg >= 0 else -1)  # 绝对值向上取整，保留符号

        rel = self.memory.get_relationship(user_id)
        stage = rel["stage"]
        affinity = rel["affinity"]
        base = STAGE_BASE_SCORES.get(stage, 0)
        scale = self.cfg.behavior.get("affinity_scale", 2)
        delta = (emotion_score + base) * scale
        if stage == "结婚" and delta < 0:
            delta = 0  # 结婚后好感度不下降
        # 喜欢/恋爱阶段封顶：可降、不可升过阈值
        cap = None
        if stage == "喜欢":
            cap = STAGE_THRESHOLDS["恋爱"]
        elif stage == "恋爱":
            cap = STAGE_THRESHOLDS["结婚"]
        if cap is not None and delta > 0:
            if affinity >= cap:
                delta = 0
            else:
                delta = min(delta, cap - affinity)

        self._clear_topic_emotions(user_id)  # 先清空，避免「已结算未清空」导致崩溃后重复结算
        new_affinity = self.memory.update_affinity(user_id, delta)
        new_stage = self._compute_stage(new_affinity, stage)
        if new_stage != stage:
            self.memory.set_stage(user_id, new_stage)
            logger.info("关系阶段变化：%s → %s（好感度 %d）", stage, new_stage, new_affinity)
        logger.info("话题好感度结算：平均 %.2f → %+d，好感 %d", avg, delta, new_affinity)
        return self.memory.get_relationship(user_id)

    def _hang_topic(self, user_id: str, topic: str, context: str = "") -> None:
        """切话题时，把旧话题挂起（含未完成上下文）。已自然收束的话题不再挂起。"""
        ctx = (context or "").strip()
        if not ctx or "无未完成" in ctx or "已收束" in ctx:
            logger.info("话题已收束，不挂起：%s", topic)
            return
        content = topic + f"（未完成：{ctx}）"
        self.memory.add_topic(user_id, "user_topic", content, importance=1)
        logger.info("挂起话题：%s", content)

    async def _maybe_settle_idle_topic(self, user_id: str) -> None:
        """静默超时的话题自动结算（老师发消息时检查）：超时则结算 + 挂起 + 清空。"""
        cur = self.memory.get_current_topic(user_id)
        if not cur["topic"]:
            return
        last = self.memory.last_message_time(user_id)
        if last is None:
            return
        idle_minutes = self.cfg.memory.get("topic_idle_minutes", 60)
        if (datetime.now() - last).total_seconds() < idle_minutes * 60:
            return  # 未超时
        if cur["kind"] == "emotion":
            await self._settle_emotion(user_id, cur["topic"])
        else:
            self._hang_topic(user_id, cur["topic"], "静默超时，未聊完")
        self._settle_topic_affinity(user_id)
        self.memory.set_current_topic(user_id, "", "normal")
        logger.info("话题静默超时已结算：%s", cur["topic"])

    async def _check_application(self, user_id: str, user_text: str, desc: str) -> bool:
        """判断老师这句话是否在明确认真地申请升级关系。"""
        prompt = (
            f"请判断老师下面这句话，是否在明确、认真地表达「想和普拉娜{desc}」的意愿。\n"
            "注意：必须是明确认真的表达（如「我们在一起吧」「嫁给我」）；开玩笑、暧昧、无关的话都不算。\n\n"
            f"老师的话：{user_text}\n\n"
            '严格只输出 JSON：{"apply": true} 或 {"apply": false}'
        )
        messages = [
            {"role": "system", "content": self._system_prompt(user_id)},
            {"role": "user", "content": prompt},
        ]
        raw = await self._chat_with_fallback(messages)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return False
        try:
            data = json.loads(m.group(0))
        except Exception:
            return False
        apply = bool(data.get("apply"))
        logger.info("申请判定：%s", "是" if apply else "否")
        return apply

    async def _handle_application(self, user_id: str, user_text: str) -> str | None:
        """判断申请，返回 'reject'/'恋爱'/'结婚'/None（**不修改状态**，升级在 _apply_upgrade）。"""
        rel = self.memory.get_relationship(user_id)
        stage, affinity = rel["stage"], rel["affinity"]
        if stage == "结婚":
            return None  # 关系已锁定不再变动，无需判断
        # 常驻判断：恋爱阶段判断「求婚」，其余（陌生/熟悉/喜欢）判断「表白」
        if stage == "恋爱":
            desc, threshold, target = "结婚", STAGE_THRESHOLDS["结婚"], "结婚"
        else:
            desc, threshold, target = "成为恋人", STAGE_THRESHOLDS["恋爱"], "恋爱"
        if not await self._check_application(user_id, user_text, desc):
            return None
        if affinity < threshold:
            return "reject"
        return target

    def _apply_upgrade(self, user_id: str, target: str) -> None:
        """阶段4：申请命中升级，修改关系状态。"""
        today = datetime.now().strftime("%Y-%m-%d")
        if target == "恋爱":
            rel = self.memory.get_relationship(user_id)
            self.memory.set_stage(user_id, "恋爱")
            self.memory.update_affinity(user_id, 80 - rel["affinity"])  # 拉到 80 热恋期
            self.memory.set_milestone(user_id, "love_date", today)
            logger.info("❤️ 关系升级：喜欢 → 恋爱（好感度 80）")
        elif target == "结婚":
            self.memory.set_stage(user_id, "结婚")
            self.memory.set_milestone(user_id, "marriage_date", today)
            logger.info("💍 关系升级：恋爱 → 结婚")

    async def regenerate_after_upgrade(self, user_id: str, user_text: str, upgraded_to: str) -> tuple[str, str] | None:
        """申请升级后，用「特殊时刻」语气重新生成回复（含自检）。"""
        special = SPECIAL_MOMENTS.get(upgraded_to)
        if not special:
            return None
        display_text, emotion = await self.generate(user_id, user_text, special_hint=special)
        try:
            display_text, emotion = await self.self_review(user_id, user_text, display_text, emotion)
        except Exception:
            pass
        try:
            rel = self.memory.get_relationship(user_id)
            emotion = await self._emotion_review(user_id, rel["stage"], display_text, emotion)
        except Exception:
            pass
        return display_text, emotion

    async def _emotion_review(self, user_id: str, stage: str, display_text: str, emotion: str) -> str:
        """独立的情感自检：情感标签是否符合当前关系阶段，不合适则修正。"""
        prompt = (
            f"普拉娜当前与老师的关系阶段是「{stage}」。她刚回复了一句话，情感标签是 [{emotion}]。\n"
            "请判断这个情感标签是否符合当前关系阶段：\n"
            "- 陌生/熟悉阶段不应输出 love 等过度亲密的标签\n"
            "- 恋爱/结婚阶段可以更亲昵\n"
            "- 不符合就给出更合适的标签，符合就原样输出\n\n"
            f"回复内容：{display_text}\n当前标签：{emotion}\n\n"
            "只输出一个标签名（neutral/sheepish/love/sulky/jealous/expectant/dumbfounded/delighted/cheerful/puzzled/like）。"
        )
        messages = [
            {"role": "system", "content": self._system_prompt(user_id)},
            {"role": "user", "content": prompt},
        ]
        raw = await self._chat_with_fallback(messages)
        new_emotion = raw.strip()
        result = new_emotion if new_emotion in EMOTION_SCORES else emotion
        if result != emotion:
            logger.info("情感自检：%s → %s", emotion, result)
        return result

    def _handle_special_emotion(self, user_id: str, emotion: str, user_text: str = "") -> None:
        """吃醋/黑化等特殊情绪挂起到话题表（高优先级，且最多只挂一个）。

        若已有「挂起」或「对话中」的吃醋事件，则不重复挂起（对话期间不产生新吃醋）。
        """
        if emotion in ("sulky", "jealous"):
            if not self.memory.has_active_emotion(user_id):
                cn = {"sulky": "吃醋", "jealous": "黑化"}.get(emotion, emotion)
                ctx = f"（老师：{user_text[:40]}）" if user_text else ""
                content = f"普拉娜{cn}{ctx}"
                self.memory.add_topic(user_id, "plana_emotion", content, importance=3)
                logger.info("挂起吃醋事件：%s", content)

    def _expire_topics(self, user_id: str, days: int = 3) -> None:
        """过期未关闭的挂起话题直接关闭（无好感度副作用；吃醋切走即结算）。"""
        for t in self.memory.expire_open_topics(user_id, days=days):
            logger.info("挂起话题已过期关闭：%s", t["content"])

    async def _judge_topic(self, user_id: str, user_text: str) -> dict:
        """①判断：一次 LLM 输出 {has_content, same, topic, old_context}。"""
        cur = self.memory.get_current_topic(user_id)
        if not cur["topic"]:
            # 首次对话，无当前话题：视为新话题，直接进入
            return {"has_content": True, "same": True, "topic": user_text[:40], "old_context": ""}

        recent = self.memory.recent(user_id, 3)
        # 阶段0已把老师这句话写入库，若它出现在历史末尾则去掉，避免与下面的「老师这句话」重复
        if recent and recent[-1].get("role") == "user" and recent[-1].get("content") == user_text:
            recent = recent[:-1]
        context = self._format_msgs(recent) if recent else "（无）"
        prompt = (
            "请判断老师这句话，输出四个字段：\n"
            "1. has_content：这句话是否有实际指向（「嗯嗯」「哦」「这样啊」等无营养内容 → false）\n"
            "2. same：这句话和「当前话题」是否同一个话题\n"
            "3. topic：当前话题的一句话总结，必须涵盖起因、经过、结果（衔接「当前话题」里的前提与这条新消息，别只写最新一句）\n"
            "4. old_context：仅当 same=false 时，之前那个话题的起因经过结果 + 未完成部分；若该话题已自然收束、无未完成部分，则输出空字符串\n\n"
            f"当前话题：{cur['topic']}（类型：{cur['kind']}）\n"
            f"老师这句话：{user_text}\n\n"
            f"【最近的对话】\n{context}\n\n"
            '严格只输出 JSON：{"has_content": true|false, "same": true|false, "topic": "...", "old_context": "..."}'
        )
        messages = [
            {"role": "system", "content": self._system_prompt(user_id)},
            {"role": "user", "content": prompt},
        ]
        raw = await self._chat_with_fallback(messages)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return {"has_content": True, "same": True, "topic": user_text[:40], "old_context": ""}
        try:
            data = json.loads(m.group(0))
        except Exception:
            return {"has_content": True, "same": True, "topic": user_text[:40], "old_context": ""}
        result = {
            "has_content": bool(data.get("has_content", True)),
            "same": bool(data.get("same", True)),
            "topic": str(data.get("topic", user_text[:40])),
            "old_context": str(data.get("old_context", "")),
        }
        logger.info("话题判断：has_content=%s same=%s topic=%s", result["has_content"], result["same"], result["topic"])
        return result

    async def _topic_decision(self, user_id: str, user_text: str, display_text: str) -> str:
        """判断话题是否结束：finished / continuing（二态，切话题由①前置判断处理）。"""
        cur = self.memory.get_current_topic(user_id)
        prompt = (
            "根据下面这轮对话，判断普拉娜刚回复完的这个话题是否已经结束：\n"
            "- finished：话题已经自然聊完、收尾了\n"
            "- continuing：话题明显还没结束，老师还想继续聊它，普拉娜适合主动续一句\n\n"
            f"当前话题：{cur['topic'] or '（无）'}\n"
            f"老师：{user_text}\n普拉娜：{display_text}\n\n"
            '严格只输出 JSON：{"state": "finished"|"continuing"}'
        )
        messages = [
            {"role": "system", "content": self._system_prompt(user_id)},
            {"role": "user", "content": prompt},
        ]
        raw = await self._chat_with_fallback(messages)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return "finished"
        try:
            data = json.loads(m.group(0))
        except Exception:
            return "finished"
        state = data.get("state", "finished")
        if state not in ("finished", "continuing"):
            state = "finished"
        logger.info("话题判定：%s", state)
        return state

    async def _topic_continuation(self, user_id: str, topic: str, user_text: str, display_text: str) -> str | None:
        """生成一句自然的话题续接（带上文，避免与主回复重复或时态错乱）。"""
        prompt = (
            f"你们正在聊「{topic}」，这个话题还没结束。请主动、自然地把话题往下续一句"
            "（可以是补充、感慨、或轻轻带出下一个相关点），15 字以内，符合普拉娜克制的性格，不要带情感标签。\n"
            "注意：\n"
            "- 不要重复普拉娜刚才已经说过或问过的内容\n"
            "- 如果普拉娜刚才已经问过老师问题（比如「今天过得怎么样」），续句就不要再反问（如「你呢」「老师呢」）或再抛一个新问题，改成补充、感慨或收尾的小动作\n"
            "- 老师正在做、或刚说要做的事，不要用「有…吗」「已经…了吗」这类完成时去问\n"
            "- 时态要符合当下：还没发生的事用将来/进行，已经发生的事才用完成\n\n"
            f"老师刚说：{user_text}\n"
            f"普拉娜刚回复：{display_text}\n\n"
            "只输出要续的那一句。"
        )
        messages = [
            {"role": "system", "content": self._system_prompt(user_id)},
            {"role": "user", "content": prompt},
        ]
        raw = await self._chat_with_fallback(messages)
        text = strip_emotion_tag(raw).strip()
        return text or None

    # ── 挂起话题拾起（finished 后）────────────────────────────────────────
    async def _can_chat(self, user_id: str) -> bool:
        """判断老师现在是否方便继续聊天（只依据最近原话，不臆测）。"""
        recent = self.memory.recent(user_id, 3)
        context = self._format_msgs(recent) if recent else "（无）"
        prompt = (
            "请判断老师现在是否方便继续和普拉娜聊天。只依据老师最近的原话判断，不要臆测：\n"
            "- 老师明确说了「要睡/睡了/晚安/去忙/开会/有事/先不聊/回头说」等 → 不方便（false）\n"
            "- 普通聊天、没说离开 → 方便（true）\n\n"
            f"【最近对话】\n{context}\n\n"
            '严格只输出 JSON：{"chat": true} 或 {"chat": false}'
        )
        messages = [
            {"role": "system", "content": self._system_prompt(user_id)},
            {"role": "user", "content": prompt},
        ]
        raw = await self._chat_with_fallback(messages)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return False
        try:
            data = json.loads(m.group(0))
        except Exception:
            return False
        return bool(data.get("chat"))

    async def _topic_suitability(self, user_id: str, topic_content: str, created_at: str) -> str:
        """判断挂起话题现在是否适合提起：suitable / stale / not_now。"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        recent = self.memory.recent(user_id, 3)
        context = self._format_msgs(recent) if recent else "（无）"
        prompt = (
            "普拉娜之前挂起了一个未聊完的话题，现在想判断是否适合主动提起。\n"
            "请综合时效性和情景氛围判断：\n"
            "- suitable：话题仍有意义、现在提起来自然\n"
            "- stale：话题已失效/过时（事情已结束、明显是过去式），再提很突兀\n"
            "- not_now：话题还有效，但当前氛围/时机不适合（如老师正忙、情绪不对），先保留\n\n"
            f"当前时间：{now}\n"
            f"话题挂起时间：{created_at}\n"
            f"话题内容：{topic_content}\n\n"
            f"【老师最近的对话】\n{context}\n\n"
            '严格只输出 JSON：{"state": "suitable"|"stale"|"not_now"}'
        )
        messages = [
            {"role": "system", "content": self._system_prompt(user_id)},
            {"role": "user", "content": prompt},
        ]
        raw = await self._chat_with_fallback(messages)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return "not_now"
        try:
            data = json.loads(m.group(0))
        except Exception:
            return "not_now"
        state = data.get("state", "not_now")
        return state if state in ("suitable", "stale", "not_now") else "not_now"

    async def _resume_topic_text(self, user_id: str, topic_content: str) -> str | None:
        """生成一句自然提起挂起话题的文案（15 字左右，不带情感标签）。"""
        prompt = (
            f"普拉娜之前和老师聊到一个话题没聊完，现在想自然地重新提起。\n"
            f"请生成一句自然、克制的提起文案（15 字左右，符合普拉娜性格，不要带情感标签）。\n\n"
            f"挂起的话题：{topic_content}\n\n"
            "只输出文案。"
        )
        messages = [
            {"role": "system", "content": self._system_prompt(user_id)},
            {"role": "user", "content": prompt},
        ]
        raw = await self._chat_with_fallback(messages)
        text = strip_emotion_tag(raw).strip()
        return text or None

    async def _resume_pending_topic(self, user_id: str) -> str | None:
        """话题 finished 后，按三层优先级拾起挂起话题；拾起后成为 current_topic。"""
        if not await self._can_chat(user_id):
            return None
        fresh_hours = self.cfg.memory.get("topic_fresh_hours", 1)
        for t in self.memory.get_open_topics(user_id, limit=3, fresh_hours=fresh_hours):
            state = await self._topic_suitability(user_id, t["content"], t.get("created_at", ""))
            if state == "suitable":
                is_emotion = t["type"] == "plana_emotion"
                text = await self._resume_emotion_text(user_id, t["content"]) if is_emotion \
                    else await self._resume_topic_text(user_id, t["content"])
                if text:
                    kind = "emotion" if is_emotion else "normal"
                    # 普通话题：current_topic 记拾起文案（而非旧话题内容），否则老师回应拾起文案时会被误判为切话题
                    topic = t["content"] if is_emotion else clean_for_tts(text)
                    self.memory.set_current_topic(user_id, topic, kind)
                    self.memory.close_topic(t["id"])
                    logger.info("拾起%s：%s", "吃醋事件" if is_emotion else "挂起话题", t["content"])
                    return text
            elif state == "stale":
                self.memory.close_topic(t["id"])
                logger.info("挂起话题已过时，关闭：%s", t["content"])
            # not_now → 保留，继续下一条
        return None

    async def _resume_emotion_text(self, user_id: str, emotion_content: str) -> str | None:
        """生成一句自然提起吃醋/黑化的文案（带一点小情绪，不带情感标签）。"""
        prompt = (
            f"普拉娜之前因为「{emotion_content}」对老师有点小情绪（吃醋/黑化），一直没消。\n"
            "请生成一句自然、克制地提起这件事的文案（15 字左右，符合普拉娜性格，"
            "可以带一点小情绪，不要带情感标签）。\n\n"
            "只输出文案。"
        )
        messages = [
            {"role": "system", "content": self._system_prompt(user_id)},
            {"role": "user", "content": prompt},
        ]
        raw = await self._chat_with_fallback(messages)
        text = strip_emotion_tag(raw).strip()
        return text or None

    async def _judge_soothing(self, user_id: str, emotion_content: str) -> bool:
        """判断老师是否安抚成功了（吃醋/黑化事件对话结束后的结算）。"""
        recent = self.memory.recent(user_id, 3)
        context = self._format_msgs(recent) if recent else "（无）"
        prompt = (
            f"普拉娜之前因为「{emotion_content}」对老师有点小情绪（吃醋/黑化），这段对话刚结束。\n"
            "请判断老师刚才是否安抚成功了（哄好了、认真道歉了、让普拉娜消气了）：\n"
            "- 成功 → true\n"
            "- 失败/没安抚/敷衍 → false\n\n"
            f"【最近的对话】\n{context}\n\n"
            '严格只输出 JSON：{"soothed": true} 或 {"soothed": false}'
        )
        messages = [
            {"role": "system", "content": self._system_prompt(user_id)},
            {"role": "user", "content": prompt},
        ]
        raw = await self._chat_with_fallback(messages)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return False
        try:
            data = json.loads(m.group(0))
        except Exception:
            return False
        return bool(data.get("soothed"))

    async def _settle_emotion(self, user_id: str, emotion_content: str) -> int:
        """结算吃醋事件（老师切走吃醋话题时触发）：安抚成功 +2、失败 -10。返回 delta。"""
        success = await self._judge_soothing(user_id, emotion_content)
        delta = 2 if success else -10
        self.memory.update_affinity(user_id, delta)
        logger.info("吃醋事件结算：%s（%s，好感 %+d）", emotion_content, "安抚成功" if success else "安抚失败", delta)
        return delta

    async def _detect_reminder(self, user_id: str, user_text: str) -> str | None:
        """判断老师这句话是否触发「稍后重新提醒」（要加班/要贪睡等）。"""
        if not any(k in user_text for k in ("加班", "睡", "困", "忙", "赶", "熬夜", "贪睡", "赖床", "补觉", "再睡", "眯")):
            return None
        prompt = (
            "判断老师这句话，是否属于「普拉娜刚提醒休息/起床，但老师要加班/要贪睡」的情况，"
            "从而需要普拉娜在稍后重新提醒一次。\n"
            "若是，给出提醒内容简述（如「提醒休息」「提醒起床」）；若不是，返回 no。\n\n"
            f"老师的话：{user_text}\n\n"
            '严格只输出 JSON：{"need": true, "remind": "提醒内容"} 或 {"need": false}'
        )
        messages = [
            {"role": "system", "content": self._system_prompt(user_id)},
            {"role": "user", "content": prompt},
        ]
        raw = await self._chat_with_fallback(messages)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except Exception:
            return None
        if data.get("need"):
            return str(data.get("remind", "")).strip() or None
        return None

    async def _reminder_text(self, user_id: str, remind: str) -> str | None:
        """到期提醒时，判断是否还需要提醒 + 生成文案。"""
        prompt = (
            f"普拉娜之前挂起了一个提醒任务：「{remind}」，现在到了提醒时间。\n"
            "请判断现在是否还需要提醒（若老师可能已经在休息/已起床，就不需要），"
            "并生成一句自然、温柔的提醒（15 字以内，带情感标签 [emotion:xxx]）。\n\n"
            '严格只输出 JSON：{"need": true, "text": "提醒内容"} 或 {"need": false}'
        )
        messages = [
            {"role": "system", "content": self._system_prompt(user_id)},
            {"role": "user", "content": prompt},
        ]
        raw = await self._chat_with_fallback(messages)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except Exception:
            return None
        if not data.get("need"):
            return None
        text = str(data.get("text", "")).strip()
        return text or None

    def _holiday_today(self, now: datetime) -> str | None:
        """返回今天的节日名（无则 None）。"""
        f = Path(self.cfg.character.get("holidays_file", "plana_chat/character/holidays.txt"))
        if not f.is_absolute():
            f = Path(__file__).resolve().parent.parent / f
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            return None
        key = now.strftime("%m-%d")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in ("：", ":"):
                if sep in line:
                    d, name = line.split(sep, 1)
                    if d.strip() == key:
                        return name.strip()
                    break
        return None

    async def greet(self, user_id: str) -> ProcessResult | None:
        """上下文感知的主动问候；最近有进行中对话时返回 None（跳过）。"""
        now = datetime.now()

        # 最近 30 分钟内有对话（进行中）→ 跳过定时问候
        last_time = self.memory.last_message_time(user_id)
        if last_time is not None and (now - last_time).total_seconds() < 30 * 60:
            logger.info("跳过主动问候：最近有进行中对话 user=%s", user_id)
            return None

        hour = now.hour
        if 5 <= hour < 11:
            period = "早上"
        elif 11 <= hour < 14:
            period = "中午"
        else:
            period = "晚上"
        user_msgs = self.memory.recent_user_messages(user_id, limit=3)
        if user_msgs:
            lines = []
            for m in user_msgs:
                try:
                    t = datetime.strptime(m["created_at"], "%Y-%m-%d %H:%M:%S")
                    diff_min = int((now - t).total_seconds() / 60)
                    if diff_min < 1:
                        when = "刚刚"
                    elif diff_min < 60:
                        when = f"{diff_min} 分钟前"
                    elif diff_min < 1440:
                        when = f"{diff_min // 60} 小时前"
                    else:
                        when = f"{diff_min // 1440} 天前"
                except Exception:
                    when = "之前"
                lines.append(f"老师（{when}）：{m['content']}")
            background = "\n".join(lines)
        else:
            background = "（暂无历史，这是第一次问候）"

        meal_hours = self.cfg.greeting.get("meal_duration_hours", 2)
        hint = (
            f"现在是 {now.strftime('%H:%M')}（{period}）。请像每天定时的问候那样，给老师发一句自然的问候。\n"
            "下面是老师最近的状态，仅供你了解近况、决定要不要多关心一句：\n"
            f"{background}\n\n"
            "注意：\n"
            "- 这只是背景信息，了解近况即可，【不要逐句回应这些旧话】（不要说「那就好」之类接话）；这是新一天的定时问候，不是接着昨天的对话继续聊\n"
            "- 分辨老师最近的原话，据此调整关心方式：\n"
            "  * 明确说「睡了/晚安/睡觉」等才可演绎成轻声自语；只说「休息」则正常关心、不要演绎睡着\n"
            "  * 只有明确说「要出门/在路上/准备走」等，才关心「路上慢点/加外套」；「去吃晚饭/下班/回家」不代表现在正在出门\n"
            f"  * 「吃饭/用餐」是短时动作（默认 {meal_hours} 小时内）：提到「去吃晚饭」等，若距现已超过该时长，用完成时关心，不要用进行时\n"
            "- 若老师提到生病/不舒服/熬夜/加班等，可自然带一句关心；否则就简单问候\n"
            "问候简短自然，动作可写在括号里，末尾带情感标签 [emotion:xxx]。\n\n"
            "只输出问候内容。"
        )

        # 纪念日/节日提示（仅早上问候时；纪念日优先）
        if period == "早上":
            rel = self.memory.get_relationship(user_id)
            today_key = now.strftime("%m-%d")
            anniversary = None
            if rel.get("love_date") and rel["love_date"][5:] == today_key:
                anniversary = "和老师在一起的纪念日"
            elif rel.get("marriage_date") and rel["marriage_date"][5:] == today_key:
                anniversary = "结婚纪念日"
            holiday = self._holiday_today(now)
            if anniversary:
                hint += f"\n\n另外，今天是{anniversary}，请用更郑重、甜蜜的方式祝福。"
            elif holiday:
                hint += f"\n\n另外，今天是「{holiday}」，请顺带一句应景的节日祝福或问候。"

        open_items = self.memory.recent_open_items(user_id, limit=3)
        if open_items:
            hint += "\n\n另外老师之前提到这些还没完成的事，可自然带一句关心（不要生硬）：" + "；".join(open_items)

        messages = [
            {"role": "system", "content": self._system_prompt(user_id)},
            {"role": "user", "content": hint},
        ]
        raw = await self._chat_with_fallback(messages)

        emotion = extract_emotion(raw)
        display_text = strip_emotion_tag(raw)

        # 主动问候也自检
        try:
            display_text, emotion = await self.self_review(user_id, "", display_text, emotion)
        except Exception:
            logger.warning("问候自检失败，保留原文", exc_info=True)

        # 主动问候也存入记忆（作为 assistant 消息，带 embedding）
        await self.store_assistant(user_id, display_text, emotion)

        voice_wav = None
        if self.cfg.reply.mode in ("text_voice", "voice"):
            voice_wav = await self._voice(display_text)
        return ProcessResult(display_text=display_text, emotion=emotion, voice_wav=voice_wav)

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
            logger.debug("TTS 请求：%s", ja[:60])

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
