"""记忆编辑器：查看/修改/删除普拉娜的长期记忆。

运行前会检查 bot 是否在跑，若在跑则拒绝编辑；编辑期间持有 edit.lock，
bot 启动时会检查该锁，互斥运行。

用法见 `plana edit` 或 `python scripts/edit_memory.py -h`。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from plana_chat import lock
from plana_chat.config import load_config
from plana_chat.memory import MemoryStore


def resolve_user(cfg, user_override: str | None) -> str:
    if user_override:
        return user_override
    ids = cfg.greeting.get("chat_ids", [])
    if ids:
        return str(ids[0])
    raise SystemExit("未找到用户 id：请在 config.yaml 的 greeting.chat_ids 配置，或用 --user 指定")


def _read_text_arg(arg: str | None) -> str:
    """把摘要/原文参数解析成文本：None 或 '-' 读 stdin，@file 读文件，否则原样返回。"""
    if arg in (None, "-"):
        return sys.stdin.read().strip()
    if arg.startswith("@"):
        return Path(arg[1:]).read_text(encoding="utf-8").strip()
    return arg


def cmd_show(mem: MemoryStore, uid: str) -> None:
    rel = mem.get_relationship(uid)
    summary, summary_at = mem.get_summary_with_time(uid)
    facts = mem.list_facts(uid)
    topics = mem.get_open_topics(uid, limit=20, fresh_hours=10000)
    cur = mem.get_current_topic(uid)
    msgs = mem.list_messages(uid, limit=1000)
    print(f"用户：{uid}")
    print(f"关系：{rel['stage']}（好感度 {rel['affinity']}）"
          f" 恋爱日期={rel.get('love_date')} 结婚日期={rel.get('marriage_date')}")
    print(f"消息：共 {len(msgs)} 条（最近 1000 条窗口）")
    print(f"事实：{len(facts)} 条")
    print(f"挂起话题：{len(topics)} 个")
    print("当前话题：", cur["topic"] or "（无）", f"（{cur['kind']}）")
    print("\n── 摘要（最近整理于", summary_at or "—", "）──")
    print(summary or "（无）")


def cmd_messages(mem: MemoryStore, uid: str, n: int) -> None:
    for m in mem.list_messages(uid, limit=n):
        who = "老师" if m["role"] == "user" else "普拉娜"
        emo = f" [{m['emotion']}]" if m.get("emotion") else ""
        text = m["content"].replace("\n", " / ")
        print(f"#{m['id']} {m['created_at']} {who}{emo}: {text}")


def cmd_facts(mem: MemoryStore, uid: str) -> None:
    for f in mem.list_facts(uid):
        print(f"#{f['id']} [{f['type']}] {f['created_at']} {f['content']}")


def cmd_topics(mem: MemoryStore, uid: str) -> None:
    for t in mem.get_open_topics(uid, limit=20, fresh_hours=10000):
        print(f"#{t['id']} [{t['type']}] {t['created_at']} {t['content']}")


def main() -> None:
    p = argparse.ArgumentParser(prog="plana edit", description="编辑普拉娜的记忆")
    p.add_argument("--user", help="用户 id（默认取 config 的 greeting.chat_ids 第一个）")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show", help="记忆概览")

    sp = sub.add_parser("messages", help="最近 N 条对话原文")
    sp.add_argument("n", nargs="?", type=int, default=20)

    sp = sub.add_parser("edit-message", help="修改一条原文")
    sp.add_argument("id", type=int)
    sp.add_argument("text", nargs="?", help="新文本（缺省读 stdin）")

    sp = sub.add_parser("del-message", help="删除一条原文")
    sp.add_argument("id", type=int)

    sub.add_parser("facts", help="列出事实/未完成项")

    sp = sub.add_parser("del-fact", help="删除一条事实")
    sp.add_argument("id", type=int)

    sub.add_parser("summary", help="显示摘要")

    sp = sub.add_parser("set-summary", help="重写摘要")
    sp.add_argument("text", nargs="?", help="新摘要（缺省读 stdin，@file 读文件）")

    sub.add_parser("relationship", help="显示关系")

    sp = sub.add_parser("set-stage", help="设置关系阶段")
    sp.add_argument("stage", choices=["陌生", "熟悉", "喜欢", "恋爱", "结婚"])

    sp = sub.add_parser("set-affinity", help="设置好感度绝对值")
    sp.add_argument("value", type=int)

    sub.add_parser("topics", help="列出挂起话题")

    sp = sub.add_parser("close-topic", help="关闭挂起话题")
    sp.add_argument("id", type=int)

    sub.add_parser("clear-current", help="清空当前话题")

    args = p.parse_args()

    # 第一步：bot 在跑则拒绝编辑
    if lock.bot_is_running():
        print("错误：plana 正在运行，请先 `plana off` 再编辑记忆。", file=sys.stderr)
        sys.exit(1)

    lock.acquire_edit_lock()
    try:
        cfg = load_config()
        uid = resolve_user(cfg, args.user)
        mem = MemoryStore(cfg.memory.db_path)

        if args.cmd == "show":
            cmd_show(mem, uid)
        elif args.cmd == "messages":
            cmd_messages(mem, uid, args.n)
        elif args.cmd == "edit-message":
            text = _read_text_arg(args.text)
            ok = mem.update_message(args.id, text)
            print("已修改" if ok else f"未找到消息 #{args.id}")
        elif args.cmd == "del-message":
            ok = mem.delete_message(args.id)
            print("已删除" if ok else f"未找到消息 #{args.id}")
        elif args.cmd == "facts":
            cmd_facts(mem, uid)
        elif args.cmd == "del-fact":
            ok = mem.delete_fact(args.id)
            print("已删除" if ok else f"未找到事实 #{args.id}")
        elif args.cmd == "summary":
            summary, summary_at = mem.get_summary_with_time(uid)
            print(f"（最近整理于 {summary_at}）\n{summary or '（无）'}")
        elif args.cmd == "set-summary":
            text = _read_text_arg(args.text)
            mem.edit_summary(uid, text)
            print("摘要已更新")
        elif args.cmd == "relationship":
            rel = mem.get_relationship(uid)
            print(f"{rel['stage']}（好感度 {rel['affinity']}）"
                  f" 恋爱日期={rel.get('love_date')} 结婚日期={rel.get('marriage_date')}")
        elif args.cmd == "set-stage":
            mem.set_stage(uid, args.stage)
            print(f"阶段已设为 {args.stage}")
        elif args.cmd == "set-affinity":
            mem.set_affinity(uid, args.value)
            print(f"好感度已设为 {mem.get_relationship(uid)['affinity']}")
        elif args.cmd == "topics":
            cmd_topics(mem, uid)
        elif args.cmd == "close-topic":
            mem.close_topic(args.id)
            print(f"话题 #{args.id} 已关闭")
        elif args.cmd == "clear-current":
            mem.clear_current_topic(uid)
            print("当前话题已清空")
    finally:
        lock.release_edit_lock()


if __name__ == "__main__":
    main()
