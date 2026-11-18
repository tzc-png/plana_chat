"""每用户对话记忆（SQLite）：滚动历史 + 长期摘要 + 事实 + 向量语义检索。

- messages：全部对话原文（含 embedding 向量），不删除，支持语义检索细节
- summaries：长期记忆摘要（含 consolidated_upto 标记已整理到的消息 id）
- facts：抽取出的「事实」与「未完成事项」
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

import numpy as np

_TIME_FMT = "%Y-%m-%d %H:%M:%S"


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
            # 旧库迁移：补列
            self._ensure_column(conn, "messages", "embedding", "BLOB")
            self._ensure_column(conn, "summaries", "consolidated_upto", "INTEGER NOT NULL DEFAULT 0")
            conn.commit()

    # ── 消息 ────────────────────────────────────────────────────────────────
    def add(
        self,
        user_id: str,
        role: str,
        content: str,
        emotion: str | None = None,
        embedding: bytes | None = None,
    ) -> None:
        now = datetime.now().strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO messages(user_id, role, content, emotion, embedding, created_at) "
                "VALUES(?,?,?,?,?,?)",
                (user_id, role, content, emotion, embedding, now),
            )
            conn.commit()

    def recent(self, user_id: str, n_rounds: int) -> list[dict]:
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT role, content, emotion FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, n_rounds * 2),
            ).fetchall()
        out: list[dict] = []
        for row in reversed(rows):
            d = {"role": row["role"], "content": row["content"]}
            if row["role"] == "assistant" and row["emotion"]:
                d["content"] = f"{d['content']} [emotion:{row['emotion']}]"
            out.append(d)
        return out

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

    def update_summary(self, user_id: str, content: str) -> None:
        now = datetime.now().strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO summaries(user_id, content, consolidated_upto, updated_at) VALUES(?,?,0,?) "
                "ON CONFLICT(user_id) DO UPDATE SET content=excluded.content, updated_at=excluded.updated_at",
                (user_id, content, now),
            )
            conn.commit()

    def add_facts(self, user_id: str, items: list[str], ftype: str = "fact") -> None:
        now = datetime.now().strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            for c in items:
                conn.execute(
                    "INSERT INTO facts(user_id, type, content, created_at) VALUES(?,?,?,?)",
                    (user_id, ftype, c, now),
                )
            conn.commit()

    def recent_facts(self, user_id: str, ftype: str | None = None, limit: int = 20) -> list[str]:
        with self._lock, closing(self._connect()) as conn:
            if ftype:
                rows = conn.execute(
                    "SELECT content FROM facts WHERE user_id=? AND type=? ORDER BY id DESC LIMIT ?",
                    (user_id, ftype, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT content FROM facts WHERE user_id=? ORDER BY id DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
        return [r["content"] for r in rows]

    def recent_open_items(self, user_id: str, limit: int = 10, days: int = 3) -> list[str]:
        since = (datetime.now() - timedelta(days=days)).strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT content FROM facts WHERE user_id=? AND type='open_item' AND created_at>=? "
                "ORDER BY id DESC LIMIT ?",
                (user_id, since, limit),
            ).fetchall()
        return [r["content"] for r in rows]

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

    # ── 话题表 ──────────────────────────────────────────────────────────
    def add_topic(self, user_id: str, type_: str, content: str, importance: int = 1) -> None:
        now = datetime.now().strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO topics(user_id, type, content, status, importance, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (user_id, type_, content, "open", importance, now, now),
            )
            conn.commit()

    def get_open_topics(self, user_id: str, type_: str | None = None, limit: int = 5) -> list[dict]:
        with self._lock, closing(self._connect()) as conn:
            if type_:
                rows = conn.execute(
                    "SELECT id, type, content, importance, created_at FROM topics "
                    "WHERE user_id=? AND status='open' AND type=? ORDER BY importance DESC, id DESC LIMIT ?",
                    (user_id, type_, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, type, content, importance, created_at FROM topics "
                    "WHERE user_id=? AND status='open' ORDER BY importance DESC, id DESC LIMIT ?",
                    (user_id, limit),
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

    def mark_topic_in_progress(self, topic_id: int) -> None:
        """把话题标记为「对话中」（用于吃醋事件拾起后的结算跟踪）。"""
        now = datetime.now().strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            conn.execute("UPDATE topics SET status='in_progress', updated_at=? WHERE id=?", (now, topic_id))
            conn.commit()

    def get_in_progress_emotions(self, user_id: str, limit: int = 1) -> list[dict]:
        """查「对话中」的吃醋/黑化事件。"""
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, content FROM topics WHERE user_id=? AND type='plana_emotion' "
                "AND status='in_progress' ORDER BY importance DESC, id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [{"id": r["id"], "content": r["content"]} for r in rows]

    def has_active_emotion(self, user_id: str) -> bool:
        """是否有「挂起」或「对话中」的吃醋事件（用于抑制重复挂起）。"""
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM topics WHERE user_id=? AND type='plana_emotion' "
                "AND status IN ('open','in_progress')",
                (user_id,),
            ).fetchone()
        return row["c"] > 0

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
                "SELECT role, content, embedding FROM messages "
                "WHERE user_id=? AND embedding IS NOT NULL ORDER BY id DESC",
                (user_id,),
            ).fetchall()

        scored: list[tuple[float, str, str]] = []
        for row in rows[exclude_recent:]:
            v = np.frombuffer(row["embedding"], dtype=np.float32)
            if v.shape[0] != emb.shape[0]:
                continue
            dot = float(np.dot(emb, v))
            na = float(np.linalg.norm(emb))
            nb = float(np.linalg.norm(v))
            sim = dot / (na * nb) if na > 0 and nb > 0 else 0.0
            scored.append((sim, row["role"], row["content"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"role": r, "content": c} for _, r, c in scored[:limit]]
