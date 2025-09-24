from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

from flask import Flask, jsonify, request
from flask_cors import CORS

from utils import read_clues
from ml_suspect_model import MODEL_PATH, train_and_save, rank_labels
from db import init_db, get_conn, list_suspects as db_list_suspects, get_suspect as db_get_suspect, insert_suspect, update_suspect, delete_suspect, list_clues as db_list_clues, insert_clue, delete_clue, list_evidence, insert_evidence, update_evidence, delete_evidence, aggregate_clues_text, persist_scores


app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent.parent


init_db()


@app.get("/")
def index():
    return jsonify({
        "service": "AI Detective API",
        "endpoints": {
            "GET /api/suspects": "List suspects with current ML scores",
            "GET /api/suspects/<id>": "Get detailed suspect profile",
            "POST /api/predict_suspects": "Rank suspects from provided clues"
        }
    })


@app.get("/api/suspects")
def api_list_suspects():
    with get_conn() as conn:
        suspects = db_list_suspects(conn)
        # Aggregate all clues text for scoring
        try:
            text = aggregate_clues_text(conn)
            if not Path(MODEL_PATH).exists():
                train_and_save(BASE_DIR / "inputs" / "sample_training.json")
            ranked = rank_labels([text], top_k=len(suspects) or 3)
            score_map = {label.lower(): float(score) for label, score in ranked}
        except Exception:
            score_map = {}
        enriched = []
        for s in suspects:
            sid = s["id"].lower()
            s["score"] = score_map.get(sid, 0.0)
            enriched.append(s)
        enriched.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return jsonify(enriched)


@app.get("/api/suspects/<sid>")
def api_suspect_detail(sid: str):
    with get_conn() as conn:
        s = db_get_suspect(conn, sid.lower())
        if not s:
            return jsonify({"error": "Not found"}), 404
        # related clues
        clues = db_list_clues(conn)
        related_clues = [c["text"] for c in clues if s["name"].lower() in c["text"].lower()]
        evidence_items = list_evidence(conn, s["id"])
        s["relatedClues"] = related_clues
        s["evidence"] = evidence_items
        return jsonify(s)


@app.post("/api/suspects")
def api_create_suspect():
    data = request.get_json(force=True)
    required = ["id", "name"]
    if not all(k in data for k in required):
        return jsonify({"error": "id and name required"}), 400
    with get_conn() as conn:
        if db_get_suspect(conn, data["id"]):
            return jsonify({"error": "id exists"}), 409
        insert_suspect(conn, data["id"], data["name"], data.get("bio", ''), data.get("avatar", ''), data.get("status", 'unknown'), data.get("tags"))
        return jsonify({"ok": True}), 201


@app.patch("/api/suspects/<sid>")
def api_update_suspect(sid: str):
    data = request.get_json(force=True) or {}
    with get_conn() as conn:
        if not db_get_suspect(conn, sid):
            return jsonify({"error": "Not found"}), 404
        update_suspect(conn, sid, **data)
        return jsonify({"ok": True})


@app.delete("/api/suspects/<sid>")
def api_delete_suspect(sid: str):
    with get_conn() as conn:
        if not db_get_suspect(conn, sid):
            return jsonify({"error": "Not found"}), 404
        delete_suspect(conn, sid)
        return jsonify({"ok": True})


@app.get("/api/clues")
def api_list_clues():
    suspect_id = request.args.get("suspect_id")
    limit_param = request.args.get("limit")
    with get_conn() as conn:
        rows = db_list_clues(conn, suspect_id)
        if limit_param:
            try:
                lim = int(limit_param)
                rows = rows[:lim]
            except ValueError:
                pass
        return jsonify(rows)


@app.post("/api/clues")
def api_add_clue():
    data = request.get_json(force=True)
    text = data.get("text")
    if not text:
        return jsonify({"error": "text required"}), 400
    with get_conn() as conn:
        insert_clue(conn, text, data.get("suspect_id"))
        return jsonify({"ok": True}), 201


@app.delete("/api/clues/<int:clue_id>")
def api_delete_clue(clue_id: int):
    with get_conn() as conn:
        delete_clue(conn, clue_id)
        return jsonify({"ok": True})


@app.get("/api/evidence/<sid>")
def api_list_evidence(sid: str):
    with get_conn() as conn:
        return jsonify(list_evidence(conn, sid))


@app.post("/api/evidence/<sid>")
def api_add_evidence(sid: str):
    data = request.get_json(force=True)
    with get_conn() as conn:
        if not db_get_suspect(conn, sid):
            return jsonify({"error": "Suspect not found"}), 404
        insert_evidence(conn, sid, data.get("type", "misc"), data.get("summary", ""), float(data.get("weight", 0)))
        return jsonify({"ok": True, "evidence": list_evidence(conn, sid)}), 201


@app.patch("/api/evidence/<int:evidence_id>")
def api_update_evidence(evidence_id: int):
    data = request.get_json(force=True) or {}
    with get_conn() as conn:
        update_evidence(conn, evidence_id, **data)
        return jsonify({"ok": True})


@app.delete("/api/evidence/<int:evidence_id>")
def api_delete_evidence(evidence_id: int):
    with get_conn() as conn:
        delete_evidence(conn, evidence_id)
        return jsonify({"ok": True})


@app.post("/api/rescore")
def api_rescore():
    # Force recomputation of scores and persist timestamp
    with get_conn() as conn:
        suspects = db_list_suspects(conn)
        try:
            text = aggregate_clues_text(conn)
            if not Path(MODEL_PATH).exists():
                train_and_save(BASE_DIR / "inputs" / "sample_training.json")
            ranked = rank_labels([text], top_k=len(suspects) or 3)
            score_map = {label.lower(): float(score) for label, score in ranked}
            persist_scores(conn, score_map)
            return jsonify({"ok": True, "scores": score_map})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.post("/api/predict_suspects")
def api_predict_suspects():
    payload = request.get_json(silent=True) or {}
    clues = payload.get("clues")
    if clues is None:
        # If not provided, aggregate from DB
        with get_conn() as conn:
            text = aggregate_clues_text(conn)
    else:
        text = ' '.join(clues)
    try:
        if not Path(MODEL_PATH).exists():
            train_and_save(BASE_DIR / "inputs" / "sample_training.json")
        ranked = rank_labels([text], top_k=3)
        return jsonify([{"label": l, "score": s} for l, s in ranked])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
