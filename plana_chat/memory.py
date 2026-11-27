"""每用户对话记忆（SQLite）：滚动历史 + 长期摘要 + 事实 + 向量语义检索。

- messages：全部对话原文（含 embedding 向量），不删除，支持语义检索细节
- summaries：长期记忆摘要（含 consolidated_upto 标记已整理到的消息 id）
- facts：抽取出的「事实」与「未完成事项」
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

import numpy as np

_TIME_FMT = "%Y-%m-%d %H:%M:%S"


def relative_time(created_at: str | None, now: datetime | None = None) -> str:
    """相对时间标签：刚刚/X 分钟前/X 小时前/昨天/X 天前；无时间返回空串。"""
    if not created_at:
        return ""
    now = now or datetime.now()
    try:
        t = datetime.strptime(created_at, _TIME_FMT)
    except (ValueError, TypeError):
        return ""
    diff = (now - t).total_seconds()
    if diff < 60:
        return "刚刚"
    if diff < 3600:
        return f"{int(diff // 60)} 分钟前"
    if diff < 86400:
        return f"{int(diff // 3600)} 小时前"
    if diff < 2 * 86400:
        return "昨天"
    return f"{int(diff // 86400)} 天前"


class MemoryStore:
    def __init__(self, db_path: str):
        p = Path(db_path)
        if not p.is_absolute():
            # 相对路径统一 resolve 到项目根，避免启动目录（cwd）不同导致记忆库漂移
            p = Path(__file__).resolve().parent.parent / p
        self.db_path = str(p)
        self._lock = Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def _init_db(self) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    emotion TEXT,
                    embedding BLOB,
                    created_at TEXT NOT NULL
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON messages(user_id, id)")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS summaries (
                    user_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    consolidated_upto INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id, type, id)")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS relationship (
                    user_id TEXT PRIMARY KEY,
                    affinity INTEGER NOT NULL DEFAULT 0,
                    stage TEXT NOT NULL DEFAULT '陌生',
                    love_date TEXT,
                    marriage_date TEXT
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS energy_state (
                    user_id TEXT PRIMARY KEY,
                    energy REAL NOT NULL DEFAULT 90,
                    updated_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    importance INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_topics_user ON topics(user_id, status, id)")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS current_topic (
                    user_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'normal',
                    emotions TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL
                )"""
            )
            # 旧库迁移：补列
            self._ensure_column(conn, "messages", "embedding", "BLOB")
            self._ensure_column(conn, "summaries", "consolidated_upto", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "current_topic", "emotions", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "facts", "duration", "TEXT")
            self._ensure_column(conn, "facts", "due_hours", "REAL")
            self._ensure_column(conn, "facts", "key", "TEXT")
            self._ensure_column(conn, "facts", "superseded", "INTEGER DEFAULT 0")
            conn.commit()

    # ── 消息 ────────────────────────────────────────────────────────────────
    def add(
        self,
        user_id: str,
        role: str,
        content: str,
        emotion: str | None = None,
        embedding: bytes | None = None,
        created_at: str | None = None,
    ) -> None:
        now = created_at or datetime.now().strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO messages(user_id, role, content, emotion, embedding, created_at) "
                "VALUES(?,?,?,?,?,?)",
                (user_id, role, content, emotion, embedding, now),
            )
            conn.commit()

    def update_latest_user_embedding(self, user_id: str, embedding: bytes) -> None:
        """回填最近一条 user 消息的 embedding（阶段2 计算后补上）。"""
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "UPDATE messages SET embedding=? WHERE id=(SELECT MAX(id) FROM messages WHERE user_id=? AND role='user')",
                (embedding, user_id),
            )
            conn.commit()

    def recent(self, user_id: str, n_rounds: int) -> list[dict]:
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT role, content, emotion, created_at FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, n_rounds * 2),
            ).fetchall()
        out: list[dict] = []
        for row in reversed(rows):
            d = {"role": row["role"], "content": row["content"], "created_at": row["created_at"]}
            if row["role"] == "assistant" and row["emotion"]:
                d["content"] = f"{d['content']} [emotion:{row['emotion']}]"
            out.append(d)
        return out

    def recent_user_messages(self, user_id: str, limit: int = 3) -> list[dict]:
        """返回最近 N 条老师原话（含时间），供主动问候判断情境。"""
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT content, created_at FROM messages WHERE user_id=? AND role='user' ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [{"content": r["content"], "created_at": r["created_at"]} for r in reversed(rows)]

    def list_messages(self, user_id: str, limit: int = 20) -> list[dict]:
        """最近 N 条消息（含 id，供记忆编辑器查看）。"""
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, role, content, emotion, created_at FROM messages "
                "WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def update_message(self, message_id: int, content: str) -> bool:
        """修改一条消息原文（清空 embedding，避免与旧语义向量不一致）。返回是否命中。"""
        with self._lock, closing(self._connect()) as conn:
            cur = conn.execute(
                "UPDATE messages SET content=?, embedding=NULL WHERE id=?", (content, message_id)
            )
            conn.commit()
            return cur.rowcount > 0

    def delete_message(self, message_id: int) -> bool:
        """删除一条消息原文。返回是否命中。"""
        with self._lock, closing(self._connect()) as conn:
            cur = conn.execute("DELETE FROM messages WHERE id=?", (message_id,))
            conn.commit()
            return cur.rowcount > 0

    def last_user_time(self, user_id: str) -> datetime | None:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT created_at FROM messages WHERE user_id=? AND role='user' ORDER BY id DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return datetime.strptime(row["created_at"], _TIME_FMT)

    def last_message_time(self, user_id: str) -> datetime | None:
        """最后一条消息（任意角色）的时间。"""
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT created_at FROM messages WHERE user_id=? ORDER BY id DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return datetime.strptime(row["created_at"], _TIME_FMT)

    # ── 记忆整理进度（不删原文，只标记已整理到的位置）──────────────────
    def consolidated_upto(self, user_id: str) -> int:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT consolidated_upto FROM summaries WHERE user_id=?", (user_id,)
            ).fetchone()
        return row["consolidated_upto"] if row else 0

    def update_consolidated_upto(self, user_id: str, upto: int) -> None:
        now = datetime.now().strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO summaries(user_id, content, consolidated_upto, updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET consolidated_upto=excluded.consolidated_upto",
                (user_id, "", upto, now),
            )
            conn.commit()

    def unconsolidated(self, user_id: str, upto: int, limit: int) -> list[dict]:
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, role, content FROM messages WHERE user_id=? AND id>? ORDER BY id ASC LIMIT ?",
                (user_id, upto, limit),
            ).fetchall()
        return [{"id": r["id"], "role": r["role"], "content": r["content"]} for r in rows]

    def max_id(self, user_id: str) -> int:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute("SELECT MAX(id) AS m FROM messages WHERE user_id=?", (user_id,)).fetchone()
        return row["m"] or 0

    # ── 长期摘要 / 事实 ─────────────────────────────────────────────────────
    def get_summary(self, user_id: str) -> str | None:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute("SELECT content FROM summaries WHERE user_id=?", (user_id,)).fetchone()
        return row["content"] if row else None

    def get_summary_with_time(self, user_id: str) -> tuple[str | None, str | None]:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT content, updated_at FROM summaries WHERE user_id=?", (user_id,)
            ).fetchone()
        if row is None:
            return None, None
        return row["content"], row["updated_at"]

    def update_summary(self, user_id: str, content: str) -> None:
        now = datetime.now().strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO summaries(user_id, content, consolidated_upto, updated_at) VALUES(?,?,0,?) "
                "ON CONFLICT(user_id) DO UPDATE SET content=excluded.content, updated_at=excluded.updated_at",
                (user_id, content, now),
            )
            conn.commit()

    def edit_summary(self, user_id: str, content: str) -> None:
        """直接改写摘要内容（保留 consolidated_upto，不触发重新整理）。"""
        now = datetime.now().strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "UPDATE summaries SET content=?, updated_at=? WHERE user_id=?",
                (content, now, user_id),
            )
            conn.commit()

    def list_facts(self, user_id: str) -> list[dict]:
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, type, content, created_at FROM facts WHERE user_id=? ORDER BY id",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_fact(self, fact_id: int) -> bool:
        with self._lock, closing(self._connect()) as conn:
            cur = conn.execute("DELETE FROM facts WHERE id=?", (fact_id,))
            conn.commit()
            return cur.rowcount > 0

    def add_facts(self, user_id: str, items: list[str], ftype: str = "fact",
                  key_map: dict | None = None) -> None:
        """存事实；key_map 为 {content: key}，带 key 的事实会作废同 key 的旧条（键控退休）。"""
        now = datetime.now().strftime(_TIME_FMT)
        km = key_map or {}
        with self._lock, closing(self._connect()) as conn:
            for c in items:
                key = km.get(c)
                if key:
                    # 同 key 旧有效条作废（不删，保留"以前是那样"）
                    conn.execute(
                        "UPDATE facts SET superseded=1 "
                        "WHERE user_id=? AND type='fact' AND key=? AND superseded=0",
                        (user_id, key),
                    )
                conn.execute(
                    "INSERT INTO facts(user_id, type, content, key, created_at) VALUES(?,?,?,?,?)",
                    (user_id, ftype, c, key, now),
                )
            conn.commit()

    def add_open_items(self, user_id: str, items: list[dict]) -> None:
        """存未完成事项，带 duration（short/long/scheduled）与 due_hours。"""
        now = datetime.now().strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            for it in items:
                content = it.get("content", "")
                if not content:
                    continue
                duration = it.get("duration", "long")
                if duration not in ("short", "long", "scheduled"):
                    duration = "long"
                due_hours = it.get("due_hours")
                try:
                    due_hours = float(due_hours) if due_hours is not None else None
                except (TypeError, ValueError):
                    due_hours = None
                conn.execute(
                    "INSERT INTO facts(user_id, type, content, duration, due_hours, created_at) VALUES(?,?,?,?,?,?)",
                    (user_id, "open_item", content, duration, due_hours, now),
                )
            conn.commit()

    def recent_facts(self, user_id: str, ftype: str | None = None, limit: int = 20) -> list[dict]:
        with self._lock, closing(self._connect()) as conn:
            if ftype:
                rows = conn.execute(
                    "SELECT content, created_at FROM facts WHERE user_id=? AND type=? AND superseded=0 ORDER BY id DESC LIMIT ?",
                    (user_id, ftype, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT content, created_at FROM facts WHERE user_id=? AND superseded=0 ORDER BY id DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
        return [{"content": r["content"], "created_at": r["created_at"]} for r in rows]

    def recent_open_items(self, user_id: str, limit: int = 10, short_hours: float = 2, max_hours: float = 72) -> list[dict]:
        """未完成事项；短时项超过 short_hours、长时项超过 max_hours、定时项超过 due_hours 视为已完成。"""
        now = datetime.now()
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT content, duration, due_hours, created_at FROM facts WHERE user_id=? AND type='open_item' "
                "ORDER BY id DESC",
                (user_id,),
            ).fetchall()
        out = []
        for r in rows:
            try:
                created = datetime.strptime(r["created_at"], _TIME_FMT)
            except (ValueError, TypeError):
                continue
            duration = r["duration"]
            if duration == "short":
                expire_hours = short_hours
            elif duration == "scheduled" and r["due_hours"] is not None:
                expire_hours = float(r["due_hours"])
            else:
                expire_hours = max_hours
            if (now - created).total_seconds() > expire_hours * 3600:
                continue
            out.append({"content": r["content"], "created_at": r["created_at"]})
            if len(out) >= limit:
                break
        return out

    # ── 关系（好感度 + 阶段）────────────────────────────────────────────
    def get_relationship(self, user_id: str) -> dict:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT affinity, stage, love_date, marriage_date FROM relationship WHERE user_id=?",
                (user_id,),
            ).fetchone()
        if row is None:
            return {"affinity": 0, "stage": "陌生", "love_date": None, "marriage_date": None}
        return {
            "affinity": row["affinity"],
            "stage": row["stage"],
            "love_date": row["love_date"],
            "marriage_date": row["marriage_date"],
        }

    def get_energy(self, user_id: str) -> float:
        """精力值（0~resting）；无记录返回默认 90。"""
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT energy FROM energy_state WHERE user_id=?", (user_id,)
            ).fetchone()
        return float(row["energy"]) if row else 90.0

    def set_energy(self, user_id: str, value: float) -> None:
        now = datetime.now().strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO energy_state(user_id, energy, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET energy=excluded.energy, updated_at=excluded.updated_at",
                (user_id, value, now),
            )
            conn.commit()

    def update_affinity(self, user_id: str, delta: int) -> int:
        """好感度增减（下限 0，由 SQL MAX 保证），返回最新好感度。"""
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO relationship(user_id, affinity) VALUES(?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET affinity=MAX(0, affinity+excluded.affinity)",
                (user_id, delta),
            )
            conn.commit()
        return self.get_relationship(user_id)["affinity"]

    def set_affinity(self, user_id: str, value: int) -> None:
        """直接设置好感度绝对值（下限 0）。"""
        value = max(0, value)
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO relationship(user_id, affinity) VALUES(?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET affinity=excluded.affinity",
                (user_id, value),
            )
            conn.commit()

    def set_stage(self, user_id: str, stage: str) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO relationship(user_id, stage) VALUES(?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET stage=excluded.stage",
                (user_id, stage),
            )
            conn.commit()

    def set_milestone(self, user_id: str, field: str, date_str: str) -> None:
        """设置关系里程碑日期（field: love_date / marriage_date）。"""
        if field not in ("love_date", "marriage_date"):
            return
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                f"INSERT INTO relationship(user_id, {field}) VALUES(?,?) "
                f"ON CONFLICT(user_id) DO UPDATE SET {field}=excluded.{field}",
                (user_id, date_str),
            )
            conn.commit()

    # ── 当前话题 ──────────────────────────────────────────────────────────
    def get_current_topic(self, user_id: str) -> dict:
        """返回 {topic, kind, emotions}；无记录返回空话题。"""
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT topic, kind, emotions FROM current_topic WHERE user_id=?", (user_id,)
            ).fetchone()
        if row is None:
            return {"topic": "", "kind": "normal", "emotions": []}
        try:
            emotions = json.loads(row["emotions"])
        except Exception:
            emotions = []
        return {"topic": row["topic"], "kind": row["kind"], "emotions": emotions}

    def set_current_topic(self, user_id: str, topic: str, kind: str = "normal") -> None:
        """设置当前话题，情感累积重置为空。"""
        now = datetime.now().strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO current_topic(user_id, topic, kind, emotions, updated_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET topic=excluded.topic, kind=excluded.kind, "
                "emotions=excluded.emotions, updated_at=excluded.updated_at",
                (user_id, topic, kind, "[]", now),
            )
            conn.commit()

    def accumulate_topic_emotion(self, user_id: str, emotion: str) -> None:
        """当前话题累积一个情感标签（立即持久化）。"""
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT emotions FROM current_topic WHERE user_id=?", (user_id,)
            ).fetchone()
            if row is None:
                return
            try:
                emotions = json.loads(row["emotions"])
            except Exception:
                emotions = []
            emotions.append(emotion)
            conn.execute(
                "UPDATE current_topic SET emotions=? WHERE user_id=?",
                (json.dumps(emotions, ensure_ascii=False), user_id),
            )
            conn.commit()

    def clear_topic_emotions(self, user_id: str) -> None:
        """清空当前话题的情感累积。"""
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "UPDATE current_topic SET emotions='[]' WHERE user_id=?", (user_id,)
            )
            conn.commit()

    def clear_current_topic(self, user_id: str) -> None:
        """清空当前话题（删除记录）。"""
        with self._lock, closing(self._connect()) as conn:
            conn.execute("DELETE FROM current_topic WHERE user_id=?", (user_id,))
            conn.commit()

    # ── 话题表 ──────────────────────────────────────────────────────────
    def add_topic(self, user_id: str, type_: str, content: str, importance: int = 1,
                  created_at: str | None = None) -> None:
        """挂起一个话题。created_at 缺省为当前时刻；静默超时挂起时传入「事件实际发生时间」，
        让时效判断与三层拾起优先级用真实事件时间，而非挂起时刻。"""
        ts = created_at or datetime.now().strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO topics(user_id, type, content, status, importance, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (user_id, type_, content, "open", importance, ts, ts),
            )
            conn.commit()

    def get_open_topics(self, user_id: str, type_: str | None = None, limit: int = 5, fresh_hours: float = 1) -> list[dict]:
        """挂起话题，按三层优先级：新鲜(N小时内)→吃醋→普通，内部时间倒序。"""
        since = (datetime.now() - timedelta(hours=fresh_hours)).strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            if type_:
                rows = conn.execute(
                    "SELECT id, type, content, importance, created_at FROM topics "
                    "WHERE user_id=? AND status='open' AND type=? ORDER BY created_at DESC, id DESC LIMIT ?",
                    (user_id, type_, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, type, content, importance, created_at FROM topics "
                    "WHERE user_id=? AND status='open' "
                    "ORDER BY CASE WHEN created_at>=? THEN 0 WHEN type='plana_emotion' THEN 1 ELSE 2 END, "
                    "created_at DESC, id DESC LIMIT ?",
                    (user_id, since, limit),
                ).fetchall()
        return [
            {"id": r["id"], "type": r["type"], "content": r["content"],
             "importance": r["importance"], "created_at": r["created_at"]}
            for r in rows
        ]

    def get_open_emotions(self, user_id: str, limit: int = 5) -> list[dict]:
        """普拉娜挂起的特殊情绪（吃醋/黑化等）。"""
        return self.get_open_topics(user_id, type_="plana_emotion", limit=limit)

    def close_topic(self, topic_id: int) -> None:
        now = datetime.now().strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            conn.execute("UPDATE topics SET status='closed', updated_at=? WHERE id=?", (now, topic_id))
            conn.commit()

    def has_active_emotion(self, user_id: str) -> bool:
        """是否有活跃吃醋事件：挂起的吃醋 OR 当前话题就是吃醋。"""
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM topics WHERE user_id=? AND type='plana_emotion' AND status='open'",
                (user_id,),
            ).fetchone()
            cur = conn.execute(
                "SELECT kind FROM current_topic WHERE user_id=?", (user_id,)
            ).fetchone()
        return row["c"] > 0 or (cur is not None and cur["kind"] == "emotion")

    def expire_open_topics(self, user_id: str, days: int = 3) -> list[dict]:
        """把超过 N 天仍未关闭的 open 话题标记为 closed，并返回（供调用方处理）。"""
        since = (datetime.now() - timedelta(days=days)).strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, type, content FROM topics WHERE user_id=? AND status='open' AND created_at<?",
                (user_id, since),
            ).fetchall()
            for r in rows:
                conn.execute("UPDATE topics SET status='closed' WHERE id=?", (r["id"],))
            conn.commit()
        return [{"id": r["id"], "type": r["type"], "content": r["content"]} for r in rows]

    # ── 重新提醒 ────────────────────────────────────────────────────────
    def get_due_reminders(self, user_id: str, minutes: int = 20) -> list[dict]:
        """返回已到期的提醒任务（type='reminder' 且创建超过 N 分钟）。"""
        since = (datetime.now() - timedelta(minutes=minutes)).strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, content FROM topics WHERE user_id=? AND type='reminder' "
                "AND status='open' AND created_at<?",
                (user_id, since),
            ).fetchall()
        return [{"id": r["id"], "content": r["content"]} for r in rows]

    def cancel_reminders(self, user_id: str) -> None:
        """取消该用户所有未到期的提醒（老师已发消息）。"""
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "UPDATE topics SET status='closed' WHERE user_id=? AND type='reminder' AND status='open'",
                (user_id,),
            )
            conn.commit()

    def has_pending_reminder(self, user_id: str) -> bool:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM topics WHERE user_id=? AND type='reminder' AND status='open'",
                (user_id,),
            ).fetchone()
        return row["c"] > 0

    # ── 向量语义检索 ───────────────────────────────────────────────────────
    def search_similar(
        self,
        user_id: str,
        embedding: bytes,
        limit: int = 5,
        exclude_recent: int = 16,
    ) -> list[dict]:
        """返回与给定向量最相似的历史消息（排除最近 exclude_recent 条，避免与滚动历史重复）。"""
        emb = np.frombuffer(embedding, dtype=np.float32)
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT role, content, created_at, embedding FROM messages "
                "WHERE user_id=? AND embedding IS NOT NULL ORDER BY id DESC",
                (user_id,),
            ).fetchall()

        scored: list[tuple[float, str, str, str]] = []
        for row in rows[exclude_recent:]:
            v = np.frombuffer(row["embedding"], dtype=np.float32)
            if v.shape[0] != emb.shape[0]:
                continue
            dot = float(np.dot(emb, v))
            na = float(np.linalg.norm(emb))
            nb = float(np.linalg.norm(v))
            sim = dot / (na * nb) if na > 0 and nb > 0 else 0.0
            scored.append((sim, row["role"], row["content"], row["created_at"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"role": r, "content": c, "created_at": t} for _, r, c, t in scored[:limit]]
