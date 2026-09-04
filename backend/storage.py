import json
import sqlite3
import logging
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)


class ChatStore:
    """基于 SQLite 的会话与消息持久化（标准库实现，无外部依赖）"""

    def __init__(self, db_path="./chat_history.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _init_tables(self):
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)"
            )

    # ---------- 会话 ----------
    def create_session(self, title=""):
        session_id = uuid4().hex
        with self._conn:
            self._conn.execute(
                "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
                (session_id, title, datetime.now().isoformat(timespec="seconds")),
            )
        return session_id

    def list_sessions(self):
        rows = self._conn.execute(
            """SELECT s.id, s.title, s.created_at,
                      (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count
               FROM sessions s
               ORDER BY s.created_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def session_exists(self, session_id):
        row = self._conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row is not None

    def delete_session(self, session_id):
        """删除会话及其全部消息，返回是否删除成功"""
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
            self._conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
        return cur.rowcount > 0

    # ---------- 消息 ----------
    def add_message(self, session_id, role, content, sources=None):
        if not self.session_exists(session_id):
            raise ValueError(f"会话不存在: {session_id}")
        with self._conn:
            self._conn.execute(
                "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    role,
                    content,
                    json.dumps(sources or [], ensure_ascii=False),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def get_messages(self, session_id):
        """返回会话全部消息（含解析后的 sources）；会话不存在返回 None"""
        if not self.session_exists(session_id):
            return None
        rows = self._conn.execute(
            "SELECT id, role, content, sources, created_at FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        messages = []
        for r in rows:
            item = dict(r)
            try:
                item["sources"] = json.loads(item["sources"])
            except (json.JSONDecodeError, TypeError):
                item["sources"] = []
            messages.append(item)
        return messages

    def get_history(self, session_id, max_messages=6):
        """
        取最近 max_messages 条消息作为多轮对话上下文（按时间正序）
        会话不存在时返回空列表
        """
        if not self.session_exists(session_id):
            return []
        rows = self._conn.execute(
            """SELECT role, content FROM (
                   SELECT id, role, content FROM messages
                   WHERE session_id = ? ORDER BY id DESC LIMIT ?
               ) ORDER BY id""",
            (session_id, max_messages),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def close(self):
        self._conn.close()
