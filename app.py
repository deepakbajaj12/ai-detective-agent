"""Convenience entrypoint so you can run:  python app.py

It ensures the src/ directory is on sys.path then imports the Flask app
instance from api.py (which lives inside src/). This avoids modifying the
existing relative imports inside src/api.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api import app  # type: ignore  # api.py is in src/ and expects to be executed there.


if __name__ == "__main__":
    # You can change host/port or disable debug here if deploying.
    # Disable Flask reloader so background job registry remains in the same process.
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

from flask import Flask, render_template_string, request, redirect, url_for
from src.case_manager import list_cases, add_case, get_clues, add_clue


app = Flask(__name__)


LIST_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>AI Detective Agent - Cases</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f4f4f4; margin: 0; padding: 0; }
        .container { max-width: 900px; margin: 40px auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px #ccc; }
        h1 { color: #333; }
        .case { border-bottom: 1px solid #eee; padding: 16px 0; }
        .case:last-child { border-bottom: none; }
        .clues { margin: 10px 0 0 20px; color: #555; }
        .add-form, .clue-form { margin-top: 20px; }
        input[type=text] { padding: 8px; margin-right: 8px; border-radius: 4px; border: 1px solid #ccc; }
        button { padding: 8px 16px; border-radius: 4px; border: none; background: #007bff; color: #fff; cursor: pointer; }
        button:hover { background: #0056b3; }
        .clue-form { margin-left: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Case List</h1>
        {% for case in cases %}
        <div class="case">
            <strong>{{ case[1] }}</strong> <span style="color: #888;">(Status: {{ case[2] }}, Created: {{ case[3] }})</span>
            <div class="clues">
                <b>Clues:</b>
                <ul>
                {% for clue in clues[case[0]] %}
                    <li>{{ clue }}</li>
                {% endfor %}
                </ul>
                <form class="clue-form" method="post" action="{{ url_for('add_clue_route', case_id=case[0]) }}">
                    <input type="text" name="clue" placeholder="Add a clue..." required>
                    <button type="submit">Add Clue</button>
                </form>
            </div>
        </div>
        {% endfor %}
        <h2>Add New Case</h2>
        <form class="add-form" method="post" action="{{ url_for('add_case_route') }}">
            <input type="text" name="name" placeholder="Case Name" required>
            <input type="text" name="status" placeholder="Status" value="in-progress" required>
            <button type="submit">Add Case</button>
        </form>
    </div>
</body>
</html>
'''


@app.route('/', methods=['GET'])
def index():
    cases = list_cases()
    clues = {case[0]: get_clues(case[0]) for case in cases}
    return render_template_string(LIST_TEMPLATE, cases=cases, clues=clues)


@app.route('/add_case', methods=['POST'])
def add_case_route():
    name = request.form['name']
    status = request.form['status']
    add_case(name, status)
    return redirect(url_for('index'))

@app.route('/add_clue/<int:case_id>', methods=['POST'])
def add_clue_route(case_id):
    clue = request.form['clue']
    add_clue(case_id, clue)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
