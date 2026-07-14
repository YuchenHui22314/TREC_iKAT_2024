"""SQLite data layer for the RALI Searcher: users, tokens, sessions, turns, per-user PTKB.

Single-file DB via stdlib sqlite3 (no server). Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib
hashlib, no external dependency). A process-wide lock serializes writes so the FastAPI worker threads
(and the /activate background thread) can share one connection safely. Tune for THIS scale (a handful
of users); move to Postgres only if it ever grows.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

_PBKDF2_ITERS = 200_000


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERS)
    return f"{salt.hex()}:{dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, _ = stored.split(":")
        return secrets.compare_digest(_hash_password(password, bytes.fromhex(salt_hex)), stored)
    except Exception:
        return False


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tokens (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    utterance TEXT NOT NULL,
    response TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS user_ptkb (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    statement TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    created_at REAL NOT NULL
);
"""


class Store:
    def __init__(self, db_path: str = ":memory:", token_ttl_seconds: Optional[float] = 30 * 86400,
                 now_fn=time.time):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")       # enforce ON DELETE CASCADE
        self._lock = threading.RLock()                       # serializes ALL connection access (reads too)
        self._now_fn = now_fn
        self._token_ttl = token_ttl_seconds                  # seconds; None = tokens never expire
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _now(self) -> float:
        return self._now_fn()

    # --- users -------------------------------------------------------------- #
    def create_user(self, username: str, password: str, is_admin: bool = False) -> int:
        with self._lock:
            try:
                cur = self._conn.execute(
                    "INSERT INTO users(username, password_hash, is_admin, created_at) VALUES (?,?,?,?)",
                    (username, _hash_password(password), int(is_admin), self._now()))
                self._conn.commit()
                return cur.lastrowid
            except sqlite3.IntegrityError:
                raise ValueError(f"username {username!r} already exists")

    def verify_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if row and _verify_password(password, row["password_hash"]):
            return {"id": row["id"], "username": row["username"], "is_admin": row["is_admin"]}
        return None

    def list_users(self) -> List[Dict[str, Any]]:
        """Public user directory for the demo login screen: usernames + admin flag ONLY
        (no ids, no hashes)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT username, is_admin FROM users ORDER BY username").fetchall()
        return [dict(r) for r in rows]

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, username, is_admin FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(row) if row else None

    # --- tokens ------------------------------------------------------------- #
    def create_token(self, user_id: int) -> str:
        tok = secrets.token_hex(24)
        with self._lock:
            self._conn.execute("INSERT INTO tokens(token, user_id, created_at) VALUES (?,?,?)",
                               (tok, user_id, self._now()))
            self._conn.commit()
        return tok

    def user_for_token(self, token: str) -> Optional[int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT user_id, created_at FROM tokens WHERE token=?", (token,)).fetchone()
            if row is None:
                return None
            if self._token_ttl is not None and self._now() - row["created_at"] > self._token_ttl:
                self._conn.execute("DELETE FROM tokens WHERE token=?", (token,))   # purge expired
                self._conn.commit()
                return None
            return row["user_id"]

    def delete_token(self, token: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM tokens WHERE token=?", (token,))
            self._conn.commit()

    # --- sessions ----------------------------------------------------------- #
    def create_session(self, user_id: int, title: str = "") -> int:
        with self._lock:
            now = self._now()
            cur = self._conn.execute(
                "INSERT INTO sessions(user_id, title, created_at, updated_at) VALUES (?,?,?,?)",
                (user_id, title, now, now))
            self._conn.commit()
            return cur.lastrowid

    def list_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE user_id=? "
                "ORDER BY updated_at DESC", (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: int, user_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE id=? AND user_id=?",
                (session_id, user_id)).fetchone()
        return dict(row) if row else None

    def rename_session(self, session_id: int, user_id: int, title: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET title=?, updated_at=? WHERE id=? AND user_id=?",
                (title, self._now(), session_id, user_id))
            self._conn.commit()
            return cur.rowcount > 0

    def delete_session(self, session_id: int, user_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM sessions WHERE id=? AND user_id=?",
                                     (session_id, user_id))
            self._conn.commit()
            return cur.rowcount > 0

    # --- turns -------------------------------------------------------------- #
    def add_turn(self, session_id: int, idx: Optional[int] = None, utterance: str = "",
                 response: str = "", payload: Optional[Dict[str, Any]] = None) -> int:
        with self._lock:
            now = self._now()
            if idx is None:                        # server-allocate the next index atomically
                row = self._conn.execute(
                    "SELECT COALESCE(MAX(idx), -1) + 1 AS n FROM turns WHERE session_id=?",
                    (session_id,)).fetchone()
                idx = row["n"]
            cur = self._conn.execute(
                "INSERT INTO turns(session_id, idx, utterance, response, payload, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (session_id, idx, utterance, response, json.dumps(payload or {}), now))
            self._conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id))
            self._conn.commit()
            return cur.lastrowid

    def list_turns(self, session_id: int) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, idx, utterance, response, payload, created_at FROM turns "
                "WHERE session_id=? ORDER BY idx", (session_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    # --- per-user PTKB ------------------------------------------------------ #
    def list_ptkb(self, user_id: int) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, statement, source, created_at FROM user_ptkb WHERE user_id=? ORDER BY id",
                (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def add_ptkb(self, user_id: int, statement: str, source: str = "manual") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO user_ptkb(user_id, statement, source, created_at) VALUES (?,?,?,?)",
                (user_id, statement, source, self._now()))
            self._conn.commit()
            return cur.lastrowid

    def update_ptkb(self, ptkb_id: int, user_id: int, statement: str) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE user_ptkb SET statement=? WHERE id=? AND user_id=?",
                                     (statement, ptkb_id, user_id))
            self._conn.commit()
            return cur.rowcount > 0

    def delete_ptkb(self, ptkb_id: int, user_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM user_ptkb WHERE id=? AND user_id=?",
                                     (ptkb_id, user_id))
            self._conn.commit()
            return cur.rowcount > 0
