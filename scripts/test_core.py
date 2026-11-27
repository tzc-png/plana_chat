"""核心逻辑回归测试（不依赖外部服务，秒级完成）。

覆盖：postprocess（情感标签/清洗）、MemoryStore（路径解析/记忆/好感度）、
关系系统（基础好感度叠加、阶段升降级）。

用法：
    source .venv/bin/activate && python scripts/test_core.py
"""
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plana_chat import postprocess
from plana_chat.config import Config
from plana_chat.memory import MemoryStore
from plana_chat.pipeline import (
    COHABITATION,
    EMOTION_SCORES,
    STAGE_BASE_SCORES,
    STAGE_COHABITATION,
    STAGE_THRESHOLDS,
    PlanaPipeline,
)

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ✓ {name}")
    else:
        _failed += 1
        print(f"  ✗ {name}")


def test_postprocess() -> None:
    print("== postprocess ==")
    check("提取半角标签", postprocess.extract_emotion("[emotion:like]你好") == "like")
    check("提取全角括号标签", postprocess.extract_emotion("（emotion:love）你好") == "love")
    check("提取半角括号标签", postprocess.extract_emotion("(emotion:sheepish)你好") == "sheepish")
    check("无标签回退 neutral", postprocess.extract_emotion("没有标签的文本") == "neutral")
    check("白名单外回退 neutral", postprocess.extract_emotion("[emotion:happy]") == "neutral")
    check("擦除标签", postprocess.strip_emotion_tag("你好[emotion:like]再见") == "你好再见")
    check(
        "清洗台词（去括号动作+破折号）",
        postprocess.clean_for_tts("（轻轻点头）老师，你好。——路上小心") == "老师，你好。",
    )


def test_memory() -> None:
    print("== MemoryStore ==")
    # db_path 相对路径应 resolve 到项目根（修「数据目录漂移」）
    m = MemoryStore("data/history.db")
    check("相对 db_path resolve 到项目根", m.db_path == str(ROOT / "data" / "history.db"))

    # 用临时库测读写（绝对路径，不会被 resolve）
    tmp = tempfile.mktemp(suffix=".db")
    ms = MemoryStore(tmp)
    check("新库关系默认陌生/0", ms.get_relationship("u1") == {
        "affinity": 0, "stage": "陌生", "love_date": None, "marriage_date": None,
    })

    ms.add("u1", "user", "你好")
    ms.add("u1", "assistant", "老师好", emotion="like")
    recent = ms.recent("u1", 5)
    check("recent 保留顺序与情感后缀", (
        [r["role"] for r in recent] == ["user", "assistant"]
        and recent[0]["content"] == "你好"
        and recent[1]["content"] == "老师好 [emotion:like]"
        and all("created_at" in r for r in recent)
    ))

    a = ms.update_affinity("u1", 4)
    check("好感度累加", a == 4)
    a = ms.update_affinity("u1", -10)
    check("好感度下限为 0", a == 0)

    ms.set_stage("u1", "熟悉")
    check("set_stage 生效", ms.get_relationship("u1")["stage"] == "熟悉")

    # add_topic 支持自定义 created_at（静默超时挂起传入事件实际发生时间）
    ms.add_topic("u1", "user_topic", "中午吃饭（未完成：静默超时，未聊完）",
                 importance=1, created_at="2026-11-26 12:05:00")
    open_topics = ms.get_open_topics("u1")
    check("add_topic 接受自定义 created_at（事件时间）",
          len(open_topics) == 1 and open_topics[0]["created_at"] == "2026-11-26 12:05:00")


def test_relationship() -> None:
    print("== 关系系统 ==")
    check("STAGE_BASE_SCORES 陌生=1 熟悉=0", STAGE_BASE_SCORES == {"陌生": 1, "熟悉": 0})
    check("阈值顺序正确", STAGE_THRESHOLDS == {"陌生": 0, "熟悉": 10, "喜欢": 30, "恋爱": 60, "结婚": 100})
    check("同住状态映射正确",
          STAGE_COHABITATION == {"陌生": "未同居", "熟悉": "未同居", "喜欢": "未同居",
                                 "恋爱": "恋爱", "结婚": "结婚"})
    check("同住状态三档齐全", set(COHABITATION) == {"未同居", "恋爱", "结婚"})

    # 基础好感度叠加（话题级：陌生 +1 / 熟悉 +0，受 scale 调控）
    scale = 1
    for stage, base in STAGE_BASE_SCORES.items():
        d_neutral = (EMOTION_SCORES["neutral"] + base) * scale
        d_like = (EMOTION_SCORES["like"] + base) * scale
        check(f"{stage}+neutral delta={d_neutral:+d}", d_neutral == (base * scale))
        check(f"{stage}+like delta={d_like:+d}", d_like == ((2 + base) * scale))

    # 话题级好感度结算（累积情感 → 结算）
    cfg = Config({"behavior": {"affinity_scale": 1}})
    tmp = tempfile.mktemp(suffix=".db")
    fake = SimpleNamespace(
        memory=MemoryStore(tmp),
        cfg=cfg,
        _compute_stage=lambda affinity, stage: PlanaPipeline._compute_stage(None, affinity, stage),
    )
    fake._clear_topic_emotions = lambda uid: fake.memory.clear_topic_emotions(uid)

    for _ in range(4):
        fake.memory.set_current_topic("u1", "话题", "normal")  # 新话题，重置 emotions
        PlanaPipeline._accumulate_emotion(fake, "u1", "like")
        PlanaPipeline._settle_topic_affinity(fake, "u1")
    rel = fake.memory.get_relationship("u1")
    check("4 个陌生话题(各 like)升到熟悉 +12", rel["stage"] == "熟悉" and rel["affinity"] == 12)

    # 阶段升降级（_compute_stage 只依赖模块常量，self 传 None）
    cs = PlanaPipeline._compute_stage
    check("陌生→熟悉(10)", cs(None, 10, "陌生") == "熟悉")
    check("熟悉→喜欢(30)", cs(None, 30, "熟悉") == "喜欢")
    check("熟悉→陌生(<10)", cs(None, 9, "熟悉") == "陌生")
    check("喜欢→熟悉(<30)", cs(None, 29, "喜欢") == "熟悉")
    check("恋爱→喜欢(<60)", cs(None, 59, "恋爱") == "喜欢")
    check("恋爱不自动升结婚(需申请)", cs(None, 200, "恋爱") == "恋爱")
    check("结婚锁定不降", cs(None, 0, "结婚") == "结婚")


def test_energy() -> None:
    print("== 精力 ==")
    tmp = tempfile.mktemp(suffix=".db")
    ms = MemoryStore(tmp)
    check("精力默认 90", ms.get_energy("u1") == 90.0)

    cfg = Config({"energy": {"resting": 90, "cost": 2, "recover_per_300s": 2, "proactive_min": 60}})
    p = object.__new__(PlanaPipeline)
    p.memory = ms
    p.cfg = cfg

    ms.set_energy("u1", 90)
    p.consume_energy("u1")
    check("扣电 90→88", ms.get_energy("u1") == 88.0)
    ms.set_energy("u1", 1)
    p.consume_energy("u1")
    check("扣电下限 0", ms.get_energy("u1") == 0.0)

    # 回血：无历史 → 不变
    ms.set_energy("u1", 50)
    p.recover_energy("u1")
    check("无历史回血不变", ms.get_energy("u1") == 50.0)

    # 精力档位（无历史，回血不生效，读精确值）
    ms.set_energy("u1", 90)
    check("档位 90=充沛", p._energy_band_text("u1").startswith("精力充沛"))
    ms.set_energy("u1", 70)
    check("档位 70=正常不注入", p._energy_band_text("u1") == "")
    ms.set_energy("u1", 45)
    check("档位 45=偏低", p._energy_band_text("u1").startswith("有点累了"))
    ms.set_energy("u1", 20)
    check("档位 20=疲惫", p._energy_band_text("u1").startswith("很累了"))
    ms.set_energy("u1", 5)
    check("档位 5=枯竭", p._energy_band_text("u1").startswith("几乎没有力气"))

    # 门槛（无历史，读精确值）
    ms.set_energy("u1", 59)
    check("59<60 判定疲惫", p.energy_too_tired("u1") is True)
    ms.set_energy("u1", 60)
    check("60 不疲惫", p.energy_too_tired("u1") is False)

    # 回血：300 秒前有老师消息 → 回约 2 点（放最后，避免给上面的精确值测试引入历史）
    ms.set_energy("u1", 0)
    ms.add("u1", "user", "hi",
           created_at=(datetime.now() - timedelta(seconds=300)).strftime("%Y-%m-%d %H:%M:%S"))
    p.recover_energy("u1")
    check("300s 回血约 2 点", 1.5 <= ms.get_energy("u1") <= 3.5)


def test_fact_retirement() -> None:
    print("== 事实键控退休 ==")
    tmp = tempfile.mktemp(suffix=".db")
    ms = MemoryStore(tmp)
    ms.add_facts("u1", ["老师喜欢喝咖啡"])
    ms.add_facts("u1", ["老师住在上海"], key_map={"老师住在上海": "住址"})
    ms.add_facts("u1", ["老师搬到北京了"], key_map={"老师搬到北京了": "住址"})
    contents = [f["content"] for f in ms.recent_facts("u1", "fact", limit=10)]
    check("住址旧值被退休", "老师住在上海" not in contents)
    check("住址新值保留", "老师搬到北京了" in contents)
    check("无 key 事实不受影响", "老师喜欢喝咖啡" in contents)
    ms.add_facts("u1", ["老师的生日是 3 月 15 日"], key_map={"老师的生日是 3 月 15 日": "生日"})
    contents = [f["content"] for f in ms.recent_facts("u1", "fact", limit=10)]
    check("不同 key 共存", "老师搬到北京了" in contents and "老师的生日是 3 月 15 日" in contents)
    check("facts_by_key 返回当前住址", ms.facts_by_key("u1", "住址") == ["老师搬到北京了"])
    check("facts_by_key 返回当前生日", ms.facts_by_key("u1", "生日") == ["老师的生日是 3 月 15 日"])


def main() -> None:
    test_postprocess()
    test_memory()
    test_relationship()
    test_energy()
    test_fact_retirement()
    print(f"\n结果：{_passed} 通过，{_failed} 失败")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
