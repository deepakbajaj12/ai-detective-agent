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
        """CREATE TABLE IF NOT EXISTS cases (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    # Analyst feedback on model rankings (human-in-the-loop)
    cur.execute(
        """CREATE TABLE IF NOT EXISTS model_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            suspect_id TEXT NOT NULL REFERENCES suspects(id) ON DELETE CASCADE,
            decision TEXT NOT NULL, -- confirm | reject | uncertain
            rank_at_feedback INTEGER,
            composite_score REAL,
            ml_score REAL,
            evidence_score REAL,
            offense_boost REAL,
            case_id TEXT REFERENCES cases(id) ON DELETE CASCADE,
            clue_id INTEGER REFERENCES clues(id) ON DELETE SET NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    # Uploaded / ingested documents (e.g., PDFs)
    cur.execute(
        """CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            original_name TEXT,
            text TEXT,
            case_id TEXT REFERENCES cases(id) ON DELETE CASCADE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS suspects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            bio TEXT,
            avatar TEXT,
            status TEXT DEFAULT 'unknown',
            tags TEXT,
            case_id TEXT REFERENCES cases(id) ON DELETE CASCADE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS clues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            suspect_id TEXT REFERENCES suspects(id) ON DELETE SET NULL,
            case_id TEXT REFERENCES cases(id) ON DELETE CASCADE,
            source_type TEXT, -- manual | document | auto
            duplicate_of_id INTEGER REFERENCES clues(id) ON DELETE SET NULL,
            similarity REAL, -- similarity score to canonical if duplicate
            clue_quality REAL, -- heuristic 0..1
            annotation_label TEXT, -- relevant | irrelevant | ambiguous
            annotation_notes TEXT,
            annotation_updated_at TEXT,
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
            case_id TEXT REFERENCES cases(id) ON DELETE CASCADE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    # Allegations / suspected offenses (many per suspect)
    cur.execute(
        """CREATE TABLE IF NOT EXISTS allegations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            suspect_id TEXT NOT NULL REFERENCES suspects(id) ON DELETE CASCADE,
            offense TEXT NOT NULL,
            description TEXT,
            severity TEXT DEFAULT 'medium', -- low | medium | high
            case_id TEXT REFERENCES cases(id) ON DELETE CASCADE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    # Chunked document segments for RAG
    cur.execute(
        """CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            case_id TEXT REFERENCES cases(id) ON DELETE CASCADE,
            chunk_index INTEGER,
            text TEXT NOT NULL,
            embedding TEXT, -- JSON serialized list[float]
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    # Extracted timeline events
    cur.execute(
        """CREATE TABLE IF NOT EXISTS timeline_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT REFERENCES cases(id) ON DELETE CASCADE,
            suspect_id TEXT REFERENCES suspects(id) ON DELETE SET NULL,
            source_type TEXT, -- clue|evidence|document
            source_id INTEGER,
            event_text TEXT,
            event_time TEXT, -- original captured string
            norm_timestamp TEXT, -- ISO normalized (YYYY-MM-DDThh:mm:ss) or date-only
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    # Basic users & API tokens (simple auth layer)
    cur.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS api_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            token TEXT UNIQUE NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_used_at TEXT
        )"""
    )
    # Model versions registry (logreg / transformer)
    cur.execute(
        """CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_tag TEXT UNIQUE NOT NULL,
            model_type TEXT NOT NULL,
            path TEXT,
            role TEXT DEFAULT 'archived', -- active | shadow | archived
            metrics TEXT, -- JSON serialized metrics
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    # Snapshots of suspect ranking state
    cur.execute(
        """CREATE TABLE IF NOT EXISTS score_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            case_id TEXT,
            payload TEXT NOT NULL, -- JSON serialized list of suspects with key score fields
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    # Token-level clue attribution (per suspect, per clue) captured at scoring time
    cur.execute(
        """CREATE TABLE IF NOT EXISTS clue_attributions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            suspect_id TEXT NOT NULL,
            clue_id INTEGER NOT NULL,
            token TEXT NOT NULL,
            weight REAL NOT NULL,
            case_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(clue_id) REFERENCES clues(id) ON DELETE CASCADE,
            FOREIGN KEY(suspect_id) REFERENCES suspects(id) ON DELETE CASCADE
        )"""
    )
    # Persistent clue embeddings (optional dense vectors; stored as JSON array for portability)
    cur.execute(
        """CREATE TABLE IF NOT EXISTS clue_embeddings (
            clue_id INTEGER PRIMARY KEY REFERENCES clues(id) ON DELETE CASCADE,
            case_id TEXT REFERENCES cases(id) ON DELETE CASCADE,
            backend TEXT, -- dense | tfidf | other
            embedding TEXT, -- JSON serialized list[float]
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
    # Multi-case columns (if migrating existing DB)
    try:
        cur.execute("ALTER TABLE suspects ADD COLUMN case_id TEXT")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE clues ADD COLUMN case_id TEXT")
    except Exception:
        pass
    # New evolving columns (idempotent attempts)
    for alter in [
        "ALTER TABLE clues ADD COLUMN source_type TEXT",
        "ALTER TABLE clues ADD COLUMN duplicate_of_id INTEGER",
        "ALTER TABLE clues ADD COLUMN similarity REAL",
        "ALTER TABLE clues ADD COLUMN clue_quality REAL",
        "ALTER TABLE clues ADD COLUMN annotation_label TEXT",
        "ALTER TABLE clues ADD COLUMN annotation_notes TEXT",
        "ALTER TABLE clues ADD COLUMN annotation_updated_at TEXT"
    ]:
        try:
            cur.execute(alter)
        except Exception:
            pass
    try:
        cur.execute("ALTER TABLE evidence ADD COLUMN case_id TEXT")
    except Exception:
        pass
    # New composite scoring columns
    try:
        cur.execute("ALTER TABLE suspects ADD COLUMN composite_score REAL")
    except Exception:
        pass
    try:
        cur.execute("ALTER TABLE suspects ADD COLUMN risk_level TEXT")
    except Exception:
        pass
    # Seed default case if none
    cur.execute("SELECT COUNT(*) as c FROM cases")
    if cur.fetchone()["c"] == 0:
        cur.execute("INSERT INTO cases(id,name,description) VALUES (?,?,?)", ("default", "Default Case", "Initial default investigative case"))
        conn.commit()
    # Ensure any existing rows get default case_id
    cur.execute("UPDATE suspects SET case_id='default' WHERE case_id IS NULL")
    cur.execute("UPDATE clues SET case_id='default' WHERE case_id IS NULL")
    cur.execute("UPDATE evidence SET case_id='default' WHERE case_id IS NULL")
    conn.commit()
    # Seed suspects if empty (into default case)
    cur.execute("SELECT COUNT(*) as c FROM suspects")
    if cur.fetchone()["c"] == 0:
        seed = [
            ("alice", "Alice", "Neighbor seen near the house. Known for meticulous planning.", "https://ui-avatars.com/api/?name=Alice", "under_watch", "meticulous,planner", "default"),
            ("bob", "Bob", "Associate mentioned in a note. Possible meeting at 9 PM.", "https://ui-avatars.com/api/?name=Bob", "associate", "meeting,associate", "default"),
            ("charlie", "Charlie", "Gloves likely used. Vehicle spotted near scene.", "https://ui-avatars.com/api/?name=Charlie", "person_of_interest", "gloves,vehicle", "default"),
        ]
        cur.executemany("INSERT INTO suspects(id,name,bio,avatar,status,tags,case_id) VALUES (?,?,?,?,?,?,?)", seed)
        conn.commit()
        # Seed some example allegations
        cur.executemany(
            "INSERT INTO allegations(suspect_id, offense, description, severity, case_id) VALUES (?,?,?,?,?)",
            [
                ("alice", "Burglary Planning", "Pattern of surveillance suggesting premeditated break-in", "medium", "default"),
                ("bob", "Conspiracy", "Referenced in meeting note about target location", "low", "default"),
                ("charlie", "Property Damage", "Gloves + vehicle near vandalism site", "high", "default"),
            ]
        )
        conn.commit()
    conn.close()


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    if d.get("tags"):
        d["tags"] = [t.strip() for t in d["tags"].split(',') if t.strip()]
    else:
        d["tags"] = []
    return d


def list_suspects(conn: sqlite3.Connection, case_id: Optional[str] = None) -> List[dict]:
    cur = conn.cursor()
    if case_id:
        cur.execute("SELECT * FROM suspects WHERE case_id=? ORDER BY created_at ASC", (case_id,))
    else:
        cur.execute("SELECT * FROM suspects ORDER BY created_at ASC")
    return [row_to_dict(r) for r in cur.fetchall()]


def get_suspect(conn: sqlite3.Connection, sid: str) -> Optional[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM suspects WHERE id=?", (sid,))
    row = cur.fetchone()
    return row_to_dict(row) if row else None


def insert_suspect(conn: sqlite3.Connection, sid: str, name: str, bio: str = '', avatar: str = '', status: str = 'unknown', tags: Optional[list[str]] = None, case_id: str = 'default'):
    tags_str = ','.join(tags) if tags else None
    cur = conn.cursor()
    cur.execute("INSERT INTO suspects(id,name,bio,avatar,status,tags,case_id) VALUES (?,?,?,?,?,?,?)", (sid, name, bio, avatar, status, tags_str, case_id))
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


def list_clues(conn: sqlite3.Connection, suspect_id: Optional[str] = None, case_id: Optional[str] = None,
               hide_duplicates: bool = False, min_quality: float | None = None,
               annotation_label: str | None = None) -> List[dict]:
    cur = conn.cursor()
    clauses = []
    params: list[Any] = []
    if suspect_id:
        clauses.append("suspect_id=?")
        params.append(suspect_id)
    if case_id:
        clauses.append("case_id=?")
        params.append(case_id)
    if hide_duplicates:
        clauses.append("(duplicate_of_id IS NULL)")
    if min_quality is not None:
        clauses.append("(clue_quality IS NULL OR clue_quality >= ?)")
        params.append(min_quality)
    if annotation_label:
        clauses.append("annotation_label=?")
        params.append(annotation_label)
    base = "SELECT *, (SELECT COUNT(*) FROM clues c2 WHERE c2.duplicate_of_id = clues.id) AS duplicate_count FROM clues"
    if clauses:
        base += " WHERE " + " AND ".join(clauses)
    base += " ORDER BY id DESC"
    cur.execute(base, tuple(params))
    return [dict(r) for r in cur.fetchall()]


def _similarity(a: str, b: str) -> float:
    """Lightweight similarity ratio using SequenceMatcher (sufficient for small corpora)."""
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def _compute_clue_quality(text: str, source_type: str | None) -> float:
    """Heuristic quality scoring.
    Components:
      - Source base weight
      - Length optimal range bonus
      - Named entity / capitalized token presence
      - Numeric presence bonus
    """
    source_base = {
        'manual': 0.7,
        'document': 0.6,
        'auto': 0.5
    }.get((source_type or '').lower(), 0.55)
    import re
    tokens = re.findall(r"[A-Za-z0-9']+", text)
    length = len(tokens)
    if length == 0:
        return 0.1
    # length score peak between 5 and 40 tokens
    if 5 <= length <= 40:
        length_score = 0.2
    else:
        # diminish outside range
        length_score = max(0.0, 0.2 - abs(length - 22) * 0.003)
    caps = sum(1 for t in tokens if len(t) > 2 and t[0].isupper())
    cap_score = 0.1 if caps > 0 else 0.0
    num_score = 0.05 if any(t.isdigit() for t in tokens) else 0.0
    quality = source_base + length_score + cap_score + num_score
    return float(min(1.0, max(0.0, quality)))


def insert_clue(conn: sqlite3.Connection, text: str, suspect_id: Optional[str] = None, case_id: str = 'default', source_type: str | None = None):
    cur = conn.cursor()
    # duplicate detection (same case scope)
    cur.execute("SELECT id, text FROM clues WHERE case_id=? ORDER BY id ASC", (case_id,))
    existing = cur.fetchall()
    dup_id = None
    sim_val = None
    best_sim = 0.0
    for r in existing:
        s = _similarity(text, r['text'])
        if s > best_sim:
            best_sim = s
            dup_id = r['id']
    # threshold for near-duplicate
    if best_sim < 0.85:  # treat below threshold as unique
        dup_id = None
    else:
        sim_val = round(best_sim, 4)
    quality = _compute_clue_quality(text, source_type)
    cur.execute("INSERT INTO clues(text, suspect_id, case_id, source_type, duplicate_of_id, similarity, clue_quality) VALUES (?,?,?,?,?,?,?)",
                (text, suspect_id, case_id, source_type, dup_id, sim_val, quality))
    conn.commit()


def delete_clue(conn: sqlite3.Connection, clue_id: int):
    cur = conn.cursor()
    cur.execute("DELETE FROM clues WHERE id=?", (clue_id,))
    conn.commit()


def annotate_clue(conn: sqlite3.Connection, clue_id: int, label: str, notes: str | None = None):
    label = label.lower()
    if label not in {'relevant','irrelevant','ambiguous'}:
        raise ValueError('invalid label')
    cur = conn.cursor()
    cur.execute("UPDATE clues SET annotation_label=?, annotation_notes=?, annotation_updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (label, notes, clue_id))
    conn.commit()


def recompute_duplicates(conn: sqlite3.Connection, case_id: str, threshold: float = 0.85):
    """Re-run duplicate detection across all clues for a case (first occurrence becomes canonical)."""
    cur = conn.cursor()
    cur.execute("SELECT id, text FROM clues WHERE case_id=? ORDER BY id ASC", (case_id,))
    rows = cur.fetchall()
    canonicals: list[sqlite3.Row] = []
    mapping: list[tuple[int, int | None, float | None]] = []  # (id, dup_of, sim)
    for r in rows:
        rid = r['id']; txt = r['text']
        best_id = None; best_sim = 0.0
        for c in canonicals:
            s = _similarity(txt, c['text'])
            if s > best_sim:
                best_sim = s; best_id = c['id']
        if best_sim >= threshold:
            mapping.append((rid, best_id, round(best_sim,4)))
        else:
            canonicals.append(r)
            mapping.append((rid, None, None))
    for rid, dup, sim in mapping:
        cur.execute("UPDATE clues SET duplicate_of_id=?, similarity=? WHERE id=?", (dup, sim, rid))
    conn.commit()


def recompute_clue_quality(conn: sqlite3.Connection, case_id: str | None = None):
    cur = conn.cursor()
    if case_id:
        cur.execute("SELECT id, text, source_type FROM clues WHERE case_id=?", (case_id,))
    else:
        cur.execute("SELECT id, text, source_type FROM clues")
    rows = cur.fetchall()
    for r in rows:
        q = _compute_clue_quality(r['text'], r['source_type'])
        cur.execute("UPDATE clues SET clue_quality=? WHERE id=?", (q, r['id']))
    conn.commit()


def list_evidence(conn: sqlite3.Connection, suspect_id: str, case_id: Optional[str] = None) -> List[dict]:
    cur = conn.cursor()
    if case_id:
        cur.execute("SELECT * FROM evidence WHERE suspect_id=? AND case_id=? ORDER BY id DESC", (suspect_id, case_id))
    else:
        cur.execute("SELECT * FROM evidence WHERE suspect_id=? ORDER BY id DESC", (suspect_id,))
    return [dict(r) for r in cur.fetchall()]


# ---- Allegations helpers ----
def list_allegations(conn: sqlite3.Connection, suspect_id: Optional[str] = None, case_id: Optional[str] = None) -> List[dict]:
    cur = conn.cursor()
    base = "SELECT * FROM allegations"
    clauses = []
    params: list[Any] = []
    if suspect_id:
        clauses.append("suspect_id=?")
        params.append(suspect_id)
    if case_id:
        clauses.append("case_id=?")
        params.append(case_id)
    if clauses:
        base += " WHERE " + " AND ".join(clauses)
    base += " ORDER BY CASE severity WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END, id ASC"
    cur.execute(base, tuple(params))
    return [dict(r) for r in cur.fetchall()]


def insert_allegation(conn: sqlite3.Connection, suspect_id: str, offense: str, description: str = '', severity: str = 'medium', case_id: str = 'default'):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO allegations(suspect_id, offense, description, severity, case_id) VALUES (?,?,?,?,?)",
        (suspect_id, offense, description, severity, case_id)
    )
    conn.commit()


def delete_allegation(conn: sqlite3.Connection, allegation_id: int):
    cur = conn.cursor()
    cur.execute("DELETE FROM allegations WHERE id=?", (allegation_id,))
    conn.commit()


def insert_evidence(conn: sqlite3.Connection, suspect_id: str, type_: str, summary: str, weight: float = 0.0, case_id: str = 'default'):
    cur = conn.cursor()
    cur.execute("INSERT INTO evidence(suspect_id, type, summary, weight, case_id) VALUES (?,?,?,?,?)", (suspect_id, type_, summary, weight, case_id))
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


def aggregate_clues_text(conn: sqlite3.Connection, case_id: Optional[str] = None) -> str:
    cur = conn.cursor()
    if case_id:
        cur.execute("SELECT text FROM clues WHERE case_id=? ORDER BY id ASC", (case_id,))
    else:
        cur.execute("SELECT text FROM clues ORDER BY id ASC")
    rows = cur.fetchall()
    return ' '.join(r["text"] for r in rows)


# ---- Documents helpers ----
def insert_document(conn: sqlite3.Connection, filename: str, original_name: str, text: str, case_id: str = 'default') -> int:
    cur = conn.cursor()
    cur.execute("INSERT INTO documents(filename, original_name, text, case_id) VALUES (?,?,?,?)", (filename, original_name, text, case_id))
    conn.commit()
    return int(cur.lastrowid)


def get_document(conn: sqlite3.Connection, doc_id: int) -> Optional[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM documents WHERE id=?", (doc_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def list_documents(conn: sqlite3.Connection, case_id: Optional[str] = None) -> list[dict]:
    cur = conn.cursor()
    if case_id:
        cur.execute("SELECT * FROM documents WHERE case_id=? ORDER BY id DESC", (case_id,))
    else:
        cur.execute("SELECT * FROM documents ORDER BY id DESC")
    return [dict(r) for r in cur.fetchall()]


# ---- Cases helpers ----
def list_cases(conn: sqlite3.Connection) -> List[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM cases ORDER BY created_at ASC")
    return [dict(r) for r in cur.fetchall()]


def get_case(conn: sqlite3.Connection, case_id: str) -> Optional[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM cases WHERE id=?", (case_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def insert_case(conn: sqlite3.Connection, case_id: str, name: str, description: str = ""):
    cur = conn.cursor()
    cur.execute("INSERT INTO cases(id,name,description) VALUES (?,?,?)", (case_id, name, description))
    conn.commit()


def persist_scores(conn: sqlite3.Connection, score_map: dict[str, float]):
    cur = conn.cursor()
    for sid, score in score_map.items():
        cur.execute(
            "UPDATE suspects SET last_score=?, last_scored_at=CURRENT_TIMESTAMP WHERE lower(id)=lower(?)",
            (float(score), sid),
        )
    conn.commit()


def persist_composite_scores(conn: sqlite3.Connection, composite_map: dict[str, float], risk_map: dict[str, str]):
    """Persist composite (ml + evidence) scores and risk level.

    composite_map: sid -> composite score (0..1)
    risk_map: sid -> risk level string (High/Medium/Low)
    """
    cur = conn.cursor()
    for sid, cscore in composite_map.items():
        cur.execute(
            "UPDATE suspects SET composite_score=?, risk_level=?, updated_at=CURRENT_TIMESTAMP WHERE lower(id)=lower(?)",
            (float(cscore), risk_map.get(sid, 'unknown'), sid)
        )
    conn.commit()


# ---- Feedback helpers ----
def insert_feedback(conn: sqlite3.Connection, suspect_id: str, decision: str, rank_at_feedback: int | None, composite_score: float | None, ml_score: float | None, evidence_score: float | None, offense_boost: float | None, case_id: str = 'default', clue_id: int | None = None):
    decision = decision.lower()
    if decision not in {'confirm','reject','uncertain'}:
        raise ValueError('invalid decision')
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO model_feedback(suspect_id, decision, rank_at_feedback, composite_score, ml_score, evidence_score, offense_boost, case_id, clue_id)
            VALUES (?,?,?,?,?,?,?,?,?)""",
        (suspect_id, decision, rank_at_feedback, composite_score, ml_score, evidence_score, offense_boost, case_id, clue_id)
    )
    conn.commit()
    return int(cur.lastrowid)


# ---- Attribution helpers ----
def clear_attributions(conn: sqlite3.Connection, case_id: str, suspect_id: str | None = None):
    cur = conn.cursor()
    if suspect_id:
        cur.execute("DELETE FROM clue_attributions WHERE case_id=? AND lower(suspect_id)=lower(?)", (case_id, suspect_id))
    else:
        cur.execute("DELETE FROM clue_attributions WHERE case_id=?", (case_id,))
    conn.commit()


def insert_attribution(conn: sqlite3.Connection, suspect_id: str, clue_id: int, token: str, weight: float, case_id: str):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO clue_attributions(suspect_id, clue_id, token, weight, case_id) VALUES (?,?,?,?,?)",
        (suspect_id, clue_id, token, float(weight), case_id)
    )
    conn.commit()


def fetch_attributions(conn: sqlite3.Connection, suspect_id: str, case_id: str) -> list[dict]:
    cur = conn.cursor()
    cur.execute(
        "SELECT clue_id, token, weight FROM clue_attributions WHERE lower(suspect_id)=lower(?) AND case_id=? ORDER BY clue_id ASC, weight DESC",
        (suspect_id, case_id)
    )
    rows = [dict(r) for r in cur.fetchall()]
    return rows

# ---- Persistent Embeddings ----
def upsert_clue_embedding(conn: sqlite3.Connection, clue_id: int, case_id: str, backend: str, embedding: list[float]):
    cur = conn.cursor()
    import json as _json
    emb_json = _json.dumps(embedding)
    # Try update then insert if missing
    cur.execute("UPDATE clue_embeddings SET embedding=?, backend=?, updated_at=CURRENT_TIMESTAMP WHERE clue_id=?", (emb_json, backend, clue_id))
    if cur.rowcount == 0:
        cur.execute("INSERT INTO clue_embeddings(clue_id, case_id, backend, embedding) VALUES (?,?,?,?)", (clue_id, case_id, backend, emb_json))
    conn.commit()

def list_case_embeddings(conn: sqlite3.Connection, case_id: str) -> list[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM clue_embeddings WHERE case_id=? ORDER BY clue_id ASC", (case_id,))
    rows = [dict(r) for r in cur.fetchall()]
    import json as _json
    for r in rows:
        if r.get('embedding'):
            try:
                r['embedding'] = _json.loads(r['embedding'])
            except Exception:
                r['embedding'] = None
    return rows

# ---- RAG Document Chunks ----
def insert_document_chunk(conn: sqlite3.Connection, document_id: int, case_id: str, chunk_index: int, text: str, embedding: list[float] | None = None):
    cur = conn.cursor()
    emb_str = json.dumps(embedding) if embedding is not None else None
    cur.execute("INSERT INTO document_chunks(document_id, case_id, chunk_index, text, embedding) VALUES (?,?,?,?,?)", (document_id, case_id, chunk_index, text, emb_str))
    conn.commit()


def list_chunks(conn: sqlite3.Connection, case_id: str, limit: int = 1000) -> list[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM document_chunks WHERE case_id=? ORDER BY document_id, chunk_index LIMIT ?", (case_id, limit))
    rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        if r.get('embedding'):
            try:
                r['embedding'] = json.loads(r['embedding'])
            except Exception:
                r['embedding'] = None
    return rows


# ---- Timeline Events ----
def insert_event(conn: sqlite3.Connection, case_id: str, source_type: str, source_id: int | None, event_text: str, event_time: str | None, norm_timestamp: str | None, suspect_id: str | None = None):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO timeline_events(case_id, suspect_id, source_type, source_id, event_text, event_time, norm_timestamp) VALUES (?,?,?,?,?,?,?)",
        (case_id, suspect_id, source_type, source_id, event_text, event_time, norm_timestamp)
    )
    conn.commit()


def list_events(conn: sqlite3.Connection, case_id: str) -> list[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM timeline_events WHERE case_id=? ORDER BY norm_timestamp ASC, id ASC", (case_id,))
    return [dict(r) for r in cur.fetchall()]


def list_feedback(conn: sqlite3.Connection, case_id: str | None = None, limit: int = 100) -> list[dict]:
    cur = conn.cursor()
    if case_id:
        cur.execute("SELECT * FROM model_feedback WHERE case_id=? ORDER BY id DESC LIMIT ?", (case_id, limit))
    else:
        cur.execute("SELECT * FROM model_feedback ORDER BY id DESC LIMIT ?", (limit,))
    return [dict(r) for r in cur.fetchall()]


def feedback_stats(conn: sqlite3.Connection, case_id: str | None = None) -> dict:
    cur = conn.cursor()
    base = "SELECT decision, COUNT(*) as c FROM model_feedback"
    params: list[Any] = []
    if case_id:
        base += " WHERE case_id=?"
        params.append(case_id)
    base += " GROUP BY decision"
    cur.execute(base, tuple(params))
    rows = cur.fetchall()
    counts = {r['decision']: r['c'] for r in rows}
    total = sum(counts.values()) or 1
    confirmation_rate = counts.get('confirm', 0) / max(1, (counts.get('confirm',0)+counts.get('reject',0))) if (counts.get('confirm',0)+counts.get('reject',0))>0 else None
    # Approx precision@1 proxy: proportion of confirmations where rank_at_feedback == 0
    cur.execute("SELECT COUNT(*) as c FROM model_feedback WHERE decision='confirm'" + (" AND case_id=?" if case_id else ''), (case_id,) if case_id else ())
    conf_total = cur.fetchone()['c']
    cur.execute("SELECT COUNT(*) as c FROM model_feedback WHERE decision='confirm' AND rank_at_feedback=0" + (" AND case_id=?" if case_id else ''), (case_id,) if case_id else ())
    conf_rank1 = cur.fetchone()['c']
    precision_at_1 = (conf_rank1 / conf_total) if conf_total else None
    return {
        'counts': counts,
        'total': total if total else 0,
        'confirmation_rate': confirmation_rate,
        'precision_at_1_proxy': precision_at_1
    }


# ---- User & Auth helpers ----
def create_user(conn: sqlite3.Connection, username: str, password_hash: str, role: str = 'user') -> int:
    cur = conn.cursor()
    cur.execute("INSERT INTO users(username, password_hash, role) VALUES (?,?,?)", (username, password_hash, role))
    conn.commit()
    return int(cur.lastrowid)


def find_user(conn: sqlite3.Connection, username: str):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    return dict(row) if row else None


def get_user_by_token(conn: sqlite3.Connection, token: str):
    cur = conn.cursor()
    cur.execute("SELECT u.* FROM users u JOIN api_tokens t ON u.id=t.user_id WHERE t.token=?", (token,))
    row = cur.fetchone()
    if row:
        cur.execute("UPDATE api_tokens SET last_used_at=CURRENT_TIMESTAMP WHERE token=?", (token,))
        conn.commit()
    return dict(row) if row else None


def create_token(conn: sqlite3.Connection, user_id: int, token: str) -> int:
    cur = conn.cursor()
    cur.execute("INSERT INTO api_tokens(user_id, token) VALUES (?,?)", (user_id, token))
    conn.commit()
    return int(cur.lastrowid)


# ---- Model Version Registry ----
def insert_model_version(conn: sqlite3.Connection, version_tag: str, model_type: str, path: str | None, role: str = 'archived', metrics: dict | None = None):
    cur = conn.cursor()
    mjson = json.dumps(metrics) if metrics else None
    cur.execute("INSERT INTO model_versions(version_tag, model_type, path, role, metrics) VALUES (?,?,?,?,?)", (version_tag, model_type, path, role, mjson))
    conn.commit()


def list_model_versions(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM model_versions ORDER BY created_at DESC")
    rows = []
    for r in cur.fetchall():
        d = dict(r)
        if d.get('metrics'):
            try:
                d['metrics'] = json.loads(d['metrics'])
            except Exception:
                d['metrics'] = None
        rows.append(d)
    return rows


def get_model_version(conn: sqlite3.Connection, version_tag: str) -> dict | None:
    cur = conn.cursor()
    cur.execute("SELECT * FROM model_versions WHERE version_tag=?", (version_tag,))
    r = cur.fetchone()
    if not r:
        return None
    d = dict(r)
    if d.get('metrics'):
        try:
            d['metrics'] = json.loads(d['metrics'])
        except Exception:
            d['metrics'] = None
    return d


def get_active_model(conn: sqlite3.Connection) -> dict | None:
    cur = conn.cursor()
    cur.execute("SELECT * FROM model_versions WHERE role='active' ORDER BY created_at DESC LIMIT 1")
    r = cur.fetchone()
    return dict(r) if r else None


def get_shadow_model(conn: sqlite3.Connection) -> dict | None:
    cur = conn.cursor()
    cur.execute("SELECT * FROM model_versions WHERE role='shadow' ORDER BY created_at DESC LIMIT 1")
    r = cur.fetchone()
    return dict(r) if r else None


def set_model_role(conn: sqlite3.Connection, version_tag: str, role: str):
    cur = conn.cursor()
    cur.execute("UPDATE model_versions SET role=? WHERE version_tag=?", (role, version_tag))
    conn.commit()


def clear_role(conn: sqlite3.Connection, role: str):
    cur = conn.cursor()
    cur.execute("UPDATE model_versions SET role='archived' WHERE role=?", (role,))
    conn.commit()


def update_model_metrics(conn: sqlite3.Connection, version_tag: str, metrics: dict):
    cur = conn.cursor()
    cur.execute("UPDATE model_versions SET metrics=? WHERE version_tag=?", (json.dumps(metrics), version_tag))
    conn.commit()


# ---- Snapshot helpers ----
def insert_snapshot(conn: sqlite3.Connection, label: str | None, case_id: str, suspects: list[dict]) -> int:
    cur = conn.cursor()
    minimal = []
    for s in suspects:
        minimal.append({
            'id': s.get('id'),
            'name': s.get('name'),
            'score': s.get('score'),
            'evidence_score': s.get('evidence_score'),
            'offense_boost': s.get('offense_boost'),
            'composite_score': s.get('composite_score'),
            'risk_level': s.get('risk_level')
        })
    cur.execute("INSERT INTO score_snapshots(label, case_id, payload) VALUES (?,?,?)", (label, case_id, json.dumps(minimal)))
    conn.commit()
    return int(cur.lastrowid)


def list_snapshots(conn: sqlite3.Connection, case_id: str | None = None, limit: int = 50) -> list[dict]:
    cur = conn.cursor()
    if case_id:
        cur.execute("SELECT * FROM score_snapshots WHERE case_id=? ORDER BY id DESC LIMIT ?", (case_id, limit))
    else:
        cur.execute("SELECT * FROM score_snapshots ORDER BY id DESC LIMIT ?", (limit,))
    rows = []
    for r in cur.fetchall():
        d = dict(r)
        try:
            d['payload'] = json.loads(d['payload'])
        except Exception:
            d['payload'] = []
        rows.append(d)
    return rows


def get_snapshot(conn: sqlite3.Connection, snapshot_id: int) -> dict | None:
    cur = conn.cursor()
    cur.execute("SELECT * FROM score_snapshots WHERE id=?", (snapshot_id,))
    r = cur.fetchone()
    if not r:
        return None
    d = dict(r)
    try:
        d['payload'] = json.loads(d['payload'])
    except Exception:
        d['payload'] = []
    return d


if __name__ == "__main__":
    init_db()
    with get_conn() as c:
        print("Suspects seeded:", len(list_suspects(c)))
