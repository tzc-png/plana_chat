"""每用户对话记忆（SQLite）：滚动历史 + 长期摘要 + 事实/未完成项。"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

_TIME_FMT = "%Y-%m-%d %H:%M:%S"


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self._lock = Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    emotion TEXT,
                    created_at TEXT NOT NULL
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON messages(user_id, id)")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS summaries (
                    user_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
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
            conn.commit()

    # ── 消息 ────────────────────────────────────────────────────────────────
    def add(self, user_id: str, role: str, content: str, emotion: str | None = None) -> None:
        now = datetime.now().strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO messages(user_id, role, content, emotion, created_at) VALUES(?,?,?,?,?)",
                (user_id, role, content, emotion, now),
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

    def count(self, user_id: str) -> int:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM messages WHERE user_id=?", (user_id,)).fetchone()
        return row["c"]

    def oldest(self, user_id: str, n: int) -> list[dict]:
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE user_id=? ORDER BY id ASC LIMIT ?",
                (user_id, n),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def delete_oldest(self, user_id: str, n: int) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "DELETE FROM messages WHERE id IN "
                "(SELECT id FROM messages WHERE user_id=? ORDER BY id ASC LIMIT ?)",
                (user_id, n),
            )
            conn.commit()

    def last_user_time(self, user_id: str) -> datetime | None:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT created_at FROM messages WHERE user_id=? AND role='user' ORDER BY id DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return datetime.strptime(row["created_at"], _TIME_FMT)

    # ── 长期摘要 ────────────────────────────────────────────────────────────
    def get_summary(self, user_id: str) -> str | None:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute("SELECT content FROM summaries WHERE user_id=?", (user_id,)).fetchone()
        return row["content"] if row else None

    def update_summary(self, user_id: str, content: str) -> None:
        now = datetime.now().strftime(_TIME_FMT)
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO summaries(user_id, content, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET content=excluded.content, updated_at=excluded.updated_at",
                (user_id, content, now),
            )
            conn.commit()

    # ── 事实 / 未完成项 ─────────────────────────────────────────────────────
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
