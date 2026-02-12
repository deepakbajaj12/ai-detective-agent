import sqlite3
from datetime import datetime

DB_PATH = "cases.db"

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL
    )''')
    cur.execute('''CREATE TABLE IF NOT EXISTS clues (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id INTEGER,
        clue TEXT,
        FOREIGN KEY(case_id) REFERENCES cases(id)
    )''')
    conn.commit()
    conn.close()

# --- Case Management ---
def add_case(name, status="in-progress"):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO cases (name, status, created_at) VALUES (?, ?, ?)",
                (name, status, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def list_cases():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, name, status, created_at FROM cases")
    cases = cur.fetchall()
    conn.close()
    return cases

def delete_case(case_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM clues WHERE case_id = ?", (case_id,))
    cur.execute("DELETE FROM cases WHERE id = ?", (case_id,))
    conn.commit()
    conn.close()

def add_clue(case_id, clue):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO clues (case_id, clue) VALUES (?, ?)", (case_id, clue))
    conn.commit()
    conn.close()

def get_clues(case_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT clue FROM clues WHERE case_id=?", (case_id,))
    clues = [row[0] for row in cur.fetchall()]
    conn.close()
    return clues

def update_case_status(case_id, status):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE cases SET status=? WHERE id=?", (status, case_id))
    conn.commit()
    conn.close()

# --- Initialize DB on import ---
init_db()
