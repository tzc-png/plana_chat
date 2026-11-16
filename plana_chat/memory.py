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
        self.db_path = str(db_path)
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
