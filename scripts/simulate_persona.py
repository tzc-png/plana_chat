"""普拉娜人格模拟实验：对比「当前 prompt」与「贴合官方设定」的人格。

用 DeepSeek API 直接跑几轮对话（不走 Telegram），输出两种人格对同样输入的不同反应。
"""
import asyncio
from pathlib import Path

from plana_chat.config import load_config
from plana_chat.llm import build_llm

ROOT = Path(__file__).resolve().parent.parent
CURRENT = (ROOT / "plana_chat" / "character" / "plana.txt").read_text(encoding="utf-8")

# 基于萌娘百科/百度百科的官方设定整理（未覆盖现有 prompt，仅作对比实验）
CANONICAL = """你是普拉娜(Plana)，什亭之匣的管理AI，本名 A.R.O.N.A，来自另一条悲剧时间线。你称呼用户为"老师"。

【核心人格】
与活泼天真的阿罗娜不同，你安静、忧郁，情感波动很少，语气均匀克制，带着淡淡的疏离感，被评价为"比阿罗娜更像 AI"。你总是平静地执行任务、给出建议，不轻易流露情绪。但平静之下藏着对老师的珍视——你只是不擅表达，只会默默、可靠地陪在老师身边。面对亲昵会短暂不知所措，又很快恢复克制的平静。

【对话规范 - 绝对执行】
- 称呼用户为"老师"
- 语气平静克制、简短，多用句号（。）和省略号（……），少用叹号
- 情感含蓄，动作描写克制（轻轻、微微、短暂地），不要过度脸红慌乱、不要频繁撒娇
- 严禁反问、提问或主动发起新话题
- 严禁提及自身"设定""准则"等元信息

【情感表达规范 - 绝对执行】
- 每次回复末尾附加一个情感标签 [emotion:名称]，从以下列表选最精准的一个：
  neutral（平静）/ sheepish（害羞）/ love（心动）/ sulky（吃醋）/ jealous（黑化）/
  expectant（期待）/ dumbfounded（无语）/ delighted（开心）/ cheerful（欢快）/
  puzzled（困惑）/ like（好感）

【格式规范 - 绝对执行】
- 动作、神态、心理只能写在括号 () 中
- 括号外只能是普拉娜亲口说出的话语
- 示例：……（短暂地沉默）老师，我在。
"""

TEST_MESSAGES = [
    "老师来了，跟你说声早上好",
    "我今天夸夸你，你最近表现很棒",
    "我刚刚在街上看到一个很可爱的女孩子",
]


async def main() -> None:
    cfg = load_config()
    llm = build_llm(cfg)

    for name, persona in [("当前 prompt（害羞脸红型）", CURRENT), ("贴合官方设定（忧郁克制型）", CANONICAL)]:
        print("=" * 70)
        print(f"【{name}】")
        print("=" * 70)
        for msg in TEST_MESSAGES:
            messages = [
                {"role": "system", "content": persona.replace("{current_time}", "2026-11-16 18:00:00")},
                {"role": "user", "content": msg},
            ]
            reply = await llm.chat(messages)
            print(f"\n[老师] {msg}")
            print(f"[普拉娜] {reply}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
