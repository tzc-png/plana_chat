"""核心逻辑回归测试（不依赖外部服务，秒级完成）。

覆盖：postprocess（情感标签/清洗）、MemoryStore（路径解析/记忆/好感度）、
关系系统（基础好感度叠加、阶段升降级）。

用法：
    source .venv/bin/activate && python scripts/test_core.py
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plana_chat import postprocess
from plana_chat.config import Config
from plana_chat.memory import MemoryStore
from plana_chat.pipeline import (
    EMOTION_SCORES,
    STAGE_BASE_SCORES,
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


def test_relationship() -> None:
    print("== 关系系统 ==")
    check("STAGE_BASE_SCORES 陌生=1 熟悉=0", STAGE_BASE_SCORES == {"陌生": 1, "熟悉": 0})
    check("阈值顺序正确", STAGE_THRESHOLDS == {"陌生": 0, "熟悉": 10, "喜欢": 30, "恋爱": 60, "结婚": 100})

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


def main() -> None:
    test_postprocess()
    test_memory()
    test_relationship()
    print(f"\n结果：{_passed} 通过，{_failed} 失败")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
