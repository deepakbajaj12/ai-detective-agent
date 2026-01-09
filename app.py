"""Convenience entrypoint so you can run:  python app.py

It ensures the src/ directory is on sys.path then imports the Flask app
instance from api.py (which lives inside src/). This avoids modifying the
existing relative imports inside src/api.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load .env file

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api import app  # type: ignore  # api.py is in src/ and expects to be executed there.


if __name__ == "__main__":
    # You can change host/port or disable debug here if deploying.
    # Disable Flask reloader so background job registry remains in the same process.
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
