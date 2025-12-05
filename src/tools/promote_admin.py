import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / 'data' / 'detective.db'

def promote(username_or_email: str):
    if not DB_PATH.exists():
        print(f"DB not found at {DB_PATH}")
        sys.exit(1)
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    # Try by email first, then username (if such column exists)
    try:
        cur.execute("UPDATE users SET role='admin' WHERE email=?", (username_or_email,))
        if cur.rowcount == 0:
            # Fallback: username column
            cur.execute("UPDATE users SET role='admin' WHERE username=?", (username_or_email,))
        conn.commit()
        if cur.rowcount > 0:
            print(f"Promoted '{username_or_email}' to admin.")
        else:
            print("No matching user found. Checked email and username.")
    except sqlite3.OperationalError as e:
        print(f"SQL error: {e}. Ensure 'users' table has 'role' column.")
    finally:
        conn.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m src.tools.promote_admin <email-or-username>")
        sys.exit(1)
    promote(sys.argv[1])
