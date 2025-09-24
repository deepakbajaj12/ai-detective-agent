from __future__ import annotations
import sqlite3
from pathlib import Path
import json
from typing import Any, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "detective.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    # Tables
    cur.execute(
        """CREATE TABLE IF NOT EXISTS suspects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            bio TEXT,
            avatar TEXT,
            status TEXT DEFAULT 'unknown',
            tags TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS clues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            suspect_id TEXT REFERENCES suspects(id) ON DELETE SET NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            suspect_id TEXT NOT NULL REFERENCES suspects(id) ON DELETE CASCADE,
            type TEXT,
            summary TEXT,
            weight REAL DEFAULT 0.0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.commit()
    # Attempt to add scoring columns if they don't exist
    try:
        cur.execute("ALTER TABLE suspects ADD COLUMN last_score REAL")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE suspects ADD COLUMN last_scored_at TEXT")
    except Exception:
        pass
    # Seed suspects if empty
    cur.execute("SELECT COUNT(*) as c FROM suspects")
    if cur.fetchone()["c"] == 0:
        seed = [
            ("alice", "Alice", "Neighbor seen near the house. Known for meticulous planning.", "https://ui-avatars.com/api/?name=Alice", "under_watch", "meticulous,planner"),
            ("bob", "Bob", "Associate mentioned in a note. Possible meeting at 9 PM.", "https://ui-avatars.com/api/?name=Bob", "associate", "meeting,associate"),
            ("charlie", "Charlie", "Gloves likely used. Vehicle spotted near scene.", "https://ui-avatars.com/api/?name=Charlie", "person_of_interest", "gloves,vehicle"),
        ]
        cur.executemany("INSERT INTO suspects(id,name,bio,avatar,status,tags) VALUES (?,?,?,?,?,?)", seed)
        conn.commit()
    conn.close()


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    if d.get("tags"):
        d["tags"] = [t.strip() for t in d["tags"].split(',') if t.strip()]
    else:
        d["tags"] = []
    return d


def list_suspects(conn: sqlite3.Connection) -> List[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM suspects ORDER BY created_at ASC")
    return [row_to_dict(r) for r in cur.fetchall()]


def get_suspect(conn: sqlite3.Connection, sid: str) -> Optional[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM suspects WHERE id=?", (sid,))
    row = cur.fetchone()
    return row_to_dict(row) if row else None


def insert_suspect(conn: sqlite3.Connection, sid: str, name: str, bio: str = '', avatar: str = '', status: str = 'unknown', tags: Optional[list[str]] = None):
    tags_str = ','.join(tags) if tags else None
    cur = conn.cursor()
    cur.execute("INSERT INTO suspects(id,name,bio,avatar,status,tags) VALUES (?,?,?,?,?,?)", (sid, name, bio, avatar, status, tags_str))
    conn.commit()


def update_suspect(conn: sqlite3.Connection, sid: str, **fields):
    allowed = {k: v for k, v in fields.items() if k in {"name", "bio", "avatar", "status", "tags"}}
    if not allowed:
        return
    if "tags" in allowed and isinstance(allowed["tags"], list):
        allowed["tags"] = ','.join(allowed["tags"])
    sets = ', '.join(f"{k}=?" for k in allowed.keys()) + ", updated_at=CURRENT_TIMESTAMP"
    params = list(allowed.values()) + [sid]
    cur = conn.cursor()
    cur.execute(f"UPDATE suspects SET {sets} WHERE id=?", params)
    conn.commit()


def delete_suspect(conn: sqlite3.Connection, sid: str):
    cur = conn.cursor()
    cur.execute("DELETE FROM suspects WHERE id=?", (sid,))
    conn.commit()


def list_clues(conn: sqlite3.Connection, suspect_id: Optional[str] = None) -> List[dict]:
    cur = conn.cursor()
    if suspect_id:
        cur.execute("SELECT * FROM clues WHERE suspect_id=? ORDER BY id DESC", (suspect_id,))
    else:
        cur.execute("SELECT * FROM clues ORDER BY id DESC")
    return [dict(r) for r in cur.fetchall()]


def insert_clue(conn: sqlite3.Connection, text: str, suspect_id: Optional[str] = None):
    cur = conn.cursor()
    cur.execute("INSERT INTO clues(text, suspect_id) VALUES (?,?)", (text, suspect_id))
    conn.commit()


def delete_clue(conn: sqlite3.Connection, clue_id: int):
    cur = conn.cursor()
    cur.execute("DELETE FROM clues WHERE id=?", (clue_id,))
    conn.commit()


def list_evidence(conn: sqlite3.Connection, suspect_id: str) -> List[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM evidence WHERE suspect_id=? ORDER BY id DESC", (suspect_id,))
    return [dict(r) for r in cur.fetchall()]


def insert_evidence(conn: sqlite3.Connection, suspect_id: str, type_: str, summary: str, weight: float = 0.0):
    cur = conn.cursor()
    cur.execute("INSERT INTO evidence(suspect_id, type, summary, weight) VALUES (?,?,?,?)", (suspect_id, type_, summary, weight))
    conn.commit()


def update_evidence(conn: sqlite3.Connection, evidence_id: int, **fields):
    allowed = {k: v for k, v in fields.items() if k in {"type", "summary", "weight"}}
    if not allowed:
        return
    sets = ', '.join(f"{k}=?" for k in allowed.keys()) + ", updated_at=CURRENT_TIMESTAMP"
    params = list(allowed.values()) + [evidence_id]
    cur = conn.cursor()
    cur.execute(f"UPDATE evidence SET {sets} WHERE id=?", params)
    conn.commit()


def delete_evidence(conn: sqlite3.Connection, evidence_id: int):
    cur = conn.cursor()
    cur.execute("DELETE FROM evidence WHERE id=?", (evidence_id,))
    conn.commit()


def aggregate_clues_text(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    cur.execute("SELECT text FROM clues ORDER BY id ASC")
    rows = cur.fetchall()
    return ' '.join(r["text"] for r in rows)


def persist_scores(conn: sqlite3.Connection, score_map: dict[str, float]):
    cur = conn.cursor()
    for sid, score in score_map.items():
        cur.execute(
            "UPDATE suspects SET last_score=?, last_scored_at=CURRENT_TIMESTAMP WHERE lower(id)=lower(?)",
            (float(score), sid),
        )
    conn.commit()


if __name__ == "__main__":
    init_db()
    with get_conn() as c:
        print("Suspects seeded:", len(list_suspects(c)))
