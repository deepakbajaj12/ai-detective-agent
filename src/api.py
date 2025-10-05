from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

from flask import Flask, jsonify, request
from flask_cors import CORS

from utils import read_clues
from ml_suspect_model import MODEL_PATH, train_and_save, rank_labels
from ml_transformer import ensure_transformer_model, predict_labels as transformer_rank
try:
    # optional: only needed for attention-based attribution
    from ml_transformer import load_transformer  # type: ignore
except Exception:  # pragma: no cover
    load_transformer = None  # type: ignore
from semantic_search import search as semantic_search, refresh as semantic_refresh, backend_mode as semantic_backend
from db import init_db, get_conn, list_suspects as db_list_suspects, get_suspect as db_get_suspect, insert_suspect, update_suspect, delete_suspect, list_clues as db_list_clues, insert_clue, delete_clue, list_evidence, insert_evidence, update_evidence, delete_evidence, aggregate_clues_text, persist_scores, persist_composite_scores, list_cases, get_case, insert_case, list_allegations, insert_allegation, delete_allegation, insert_document, list_documents, get_document, insert_feedback, list_feedback, feedback_stats, clear_attributions, insert_attribution, fetch_attributions, insert_document_chunk, list_chunks, insert_event, list_events, create_user, find_user, create_token, get_user_by_token, annotate_clue, recompute_duplicates, recompute_clue_quality
from graph_builder import build_graph
from gen_ai import generate_case_analysis


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
            "GET /api/suspects/<id>/attribution": "Token/clue attribution for a suspect (if generated)",
            "GET /api/suspects/<id>/risk_explanation": "Structured risk factor narrative",
            "POST /api/predict_suspects": "Rank suspects from provided clues",
            "GET /api/search?q=...": "Semantic/lexical search over clues",
            "POST /api/analysis": "Generate analytical case summary (AI / heuristic)",
            "POST /api/documents/upload": "Upload PDF, extract text, auto-suggest suspects",
            "GET /api/documents": "List ingested documents",
            "POST /api/feedback": "Submit analyst feedback (confirm/reject/uncertain)",
            "GET /api/feedback": "List feedback (latest)",
            "GET /api/feedback/stats": "Aggregate feedback metrics",
            "POST /api/simulate": "Ephemeral composite score what-if simulation",
            "GET /api/qa": "Hybrid RAG question answering (top chunks + clues)",
            "GET /api/timeline": "Extracted chronological events",
            "GET /api/graph": "Relationship graph JSON",
            "POST /api/auth/register": "Create a new user (initial open mode)",
            "POST /api/auth/login": "Get an API token",
            "GET /api/metrics": "System + model overview metrics"
        }
    })


# ---- Auth Utilities ----
import hashlib, secrets
from functools import wraps

def _hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode('utf-8')).hexdigest()


def auth_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization')
        if token and token.lower().startswith('bearer '):
            token = token.split(None,1)[1]
        if not token:
            return jsonify({'error': 'auth required'}), 401
        with get_conn() as conn:
            user = get_user_by_token(conn, token)
        if not user:
            return jsonify({'error': 'invalid token'}), 401
        request.user = user  # type: ignore
        return fn(*args, **kwargs)
    return wrapper


@app.post('/api/auth/register')
def api_register():
    data = request.get_json(force=True) or {}
    username = data.get('username','').strip().lower()
    password = data.get('password','')
    if not username or not password:
        return jsonify({'error': 'username and password required'}), 400
    with get_conn() as conn:
        if find_user(conn, username):
            return jsonify({'error': 'username exists'}), 409
        uid = create_user(conn, username, _hash_password(password), role='user')
        tok = secrets.token_hex(16)
        create_token(conn, uid, tok)
    return jsonify({'ok': True, 'token': tok})


@app.post('/api/auth/login')
def api_login():
    data = request.get_json(force=True) or {}
    username = data.get('username','').strip().lower()
    password = data.get('password','')
    if not username or not password:
        return jsonify({'error': 'username and password required'}), 400
    with get_conn() as conn:
        user = find_user(conn, username)
        if not user or user['password_hash'] != _hash_password(password):
            return jsonify({'error': 'invalid credentials'}), 401
        # issue new token each login (simplest strategy)
        tok = secrets.token_hex(16)
        create_token(conn, user['id'], tok)
    return jsonify({'ok': True, 'token': tok})


# ---- Helper: lightweight token weighting (heuristic) ----
def _compute_token_weights(text: str, top_n: int = 10) -> dict[str, float]:
    import re, math
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text)]
    if not tokens:
        return {}
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    # tf * log(len/df) but df==frequency here (no corpus) -> simplified weight = freq * log(1+freq)
    weights = {t: f * math.log(1 + f) for t, f in freq.items()}
    # normalize
    max_w = max(weights.values()) if weights else 1.0
    norm = {t: w / max_w for t, w in weights.items()}
    # take top_n
    return dict(sorted(norm.items(), key=lambda x: x[1], reverse=True)[:top_n])


# ---- Transformer attention based token attribution (fallback to heuristic) ----
def _attention_token_weights(text: str, tokenizer, model, top_n: int = 10) -> dict[str, float]:  # pragma: no cover (runtime only)
    try:
        import torch  # local import to avoid mandatory dependency if transformer unused
        enc = tokenizer(text, truncation=True, padding=False, max_length=128, return_tensors='pt')
        with torch.no_grad():
            out = model(**enc, output_attentions=True)
        # DistilBERT: attentions is tuple(layers) each (batch, heads, seq, seq)
        attn_last = out.attentions[-1].mean(1)  # (batch, seq, seq) averaged over heads
        cls_to_tokens = attn_last[0, 0, 1:]  # attention from CLS (position 0) to each subsequent token
        token_ids = enc['input_ids'][0, 1:]  # skip CLS
        raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        # merge wordpieces
        merged: dict[str, float] = {}
        current_word = ''
        current_score = 0.0
        for tok, score in zip(raw_tokens, cls_to_tokens.tolist()):
            if tok.startswith('##'):
                current_word += tok[2:]
                current_score += score
            else:
                if current_word:
                    merged[current_word] = max(merged.get(current_word, 0.0), current_score)
                current_word = tok
                current_score = score
        if current_word:
            merged[current_word] = max(merged.get(current_word, 0.0), current_score)
        if not merged:
            return {}
        # normalize
        max_w = max(merged.values()) or 1.0
        norm = {k: v / max_w for k, v in merged.items()}
        return dict(sorted(norm.items(), key=lambda x: x[1], reverse=True)[:top_n])
    except Exception:
        return {}


@app.get('/api/cases')
def api_list_cases():
    with get_conn() as conn:
        return jsonify(list_cases(conn))


@app.post('/api/cases')
@auth_required
def api_create_case():
    data = request.get_json(force=True) or {}
    cid = data.get('id')
    name = data.get('name')
    if not cid or not name:
        return jsonify({'error': 'id and name required'}), 400
    with get_conn() as conn:
        if get_case(conn, cid):
            return jsonify({'error': 'case id exists'}), 409
        insert_case(conn, cid, name, data.get('description',''))
        return jsonify({'ok': True}), 201


@app.get("/api/suspects")
def api_list_suspects():
    """List suspects including ML score, composite score and risk level.

    Composite score = alpha * ml_score + (1-alpha) * evidence_score
    evidence_score is normalized sum of evidence weights (capped at 1.0)
    """
    alpha_param = request.args.get('alpha')
    offense_beta_param = request.args.get('offense_beta')
    try:
        alpha = float(alpha_param) if alpha_param is not None else 0.7
    except ValueError:
        alpha = 0.7
    alpha = max(0.0, min(1.0, alpha))
    try:
        offense_beta = float(offense_beta_param) if offense_beta_param is not None else 0.1
    except ValueError:
        offense_beta = 0.1
    # keep additive influence restrained
    offense_beta = max(0.0, min(0.5, offense_beta))

    case_id = request.args.get('case_id') or 'default'
    with get_conn() as conn:
        if case_id and not get_case(conn, case_id):
            return jsonify({'error': 'case not found'}), 404
        suspects = db_list_suspects(conn, case_id)
        # fetch primary offense (highest severity) per suspect
        for s in suspects:
            allegations = list_allegations(conn, s['id'], case_id)
            # order allegations by severity (high > medium > low) then id
            severity_rank = {'high': 3, 'medium': 2, 'low': 1}
            allegations.sort(key=lambda a: (-severity_rank.get(a.get('severity','low'),1), a.get('id',0)))
            s['allegations'] = allegations  # include raw allegations for UI (limited fields acceptable)
            if allegations:
                primary = allegations[0]
                s['primary_offense'] = primary['offense']
                s['primary_offense_severity'] = primary['severity']
                s['allegation_count'] = len(allegations)
        # 1. ML scores (prefer transformer if available)
        transformer_used = False
        try:
            text = aggregate_clues_text(conn, case_id)
            # Attempt transformer model
            try:
                ensure_transformer_model(BASE_DIR / "inputs" / "sample_training.json")
                t_ranked = transformer_rank([text], top_k=len(suspects) or 3)
            except Exception:
                t_ranked = []
            if t_ranked:
                transformer_used = True
                ml_score_map = {label.lower(): float(score) for label, score in t_ranked}
            else:
                if not Path(MODEL_PATH).exists():
                    train_and_save(BASE_DIR / "inputs" / "sample_training.json")
                ranked = rank_labels([text], top_k=len(suspects) or 3)
                ml_score_map = {label.lower(): float(score) for label, score in ranked}
        except Exception:
            ml_score_map = {}
            transformer_used = False
        # 2. Evidence scores per suspect
        evidence_score_map: dict[str, float] = {}
        for s in suspects:
            evid = list_evidence(conn, s['id'], case_id)
            if evid:
                total = sum((item.get('weight') or 0.0) for item in evid)
                # simple normalization: assume 5 strong evidence items max -> cap at 1.0
                evidence_score_map[s['id'].lower()] = min(1.0, total / 5.0)
            else:
                evidence_score_map[s['id'].lower()] = 0.0
        # 3. Composite
        composite_map: dict[str, float] = {}
        risk_map: dict[str, str] = {}
        severity_value_map = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
        for s in suspects:
            sid = s['id'].lower()
            ml = ml_score_map.get(sid, 0.0)
            ev = evidence_score_map.get(sid, 0.0)
            base_comp = alpha * ml + (1-alpha) * ev
            sev = severity_value_map.get(s.get('primary_offense_severity','').lower(), 0.0)
            offense_boost = offense_beta * sev
            comp = min(1.0, base_comp + offense_boost)
            composite_map[sid] = comp
            if comp >= 0.6:
                risk_map[sid] = 'High'
            elif comp >= 0.4:
                risk_map[sid] = 'Medium'
            else:
                risk_map[sid] = 'Low'
            s['score'] = ml
            s['evidence_score'] = ev
            s['composite_base'] = base_comp
            s['offense_boost'] = offense_boost
            s['composite_score'] = comp
            s['risk_level'] = risk_map[sid]
        # Build clue attribution heuristics (optional, only if requested via ?attribution=1)
        if request.args.get('attribution') == '1':
            with get_conn() as c2:
                try:
                    clear_attributions(c2, case_id)
                except Exception:
                    pass
                clues_rows = db_list_clues(c2, case_id=case_id)
                # Attempt transformer attention for richer attribution
                attn_tokenizer = attn_model = None
                if load_transformer:
                    try:
                        attn_model, attn_tokenizer, _ = load_transformer()  # type: ignore
                    except Exception:
                        attn_model = None
                for s in suspects:
                    sid = s['id']
                    name_tokens = {tok.lower() for tok in s['name'].split()}
                    for clue in clues_rows:
                        text = clue['text']
                        if attn_model and attn_tokenizer:
                            tw = _attention_token_weights(text, attn_tokenizer, attn_model, top_n=10)
                            if not tw:
                                tw = _compute_token_weights(text, top_n=8)
                        else:
                            tw = _compute_token_weights(text, top_n=8)
                        for tok, w in tw.items():
                            weight = w * (1.3 if tok.lower() in name_tokens else 1.0)
                            try:
                                insert_attribution(c2, sid, clue['id'], tok, weight, case_id)
                            except Exception:
                                pass
        # expose the parameters used so UI can surface them
        meta = {
            'alpha': alpha,
            'offense_beta': offense_beta,
            'ml_backend': 'transformer' if transformer_used else 'logreg'
        }
        try:
            persist_composite_scores(conn, composite_map, risk_map)
        except Exception:
            pass
        suspects.sort(key=lambda x: x.get('composite_score', 0.0), reverse=True)
        return jsonify({'suspects': suspects, 'meta': meta})


@app.get("/api/suspects/<sid>")
def api_suspect_detail(sid: str):
    case_id = request.args.get('case_id') or 'default'
    with get_conn() as conn:
        s = db_get_suspect(conn, sid.lower())
        if not s:
            return jsonify({"error": "Not found"}), 404
        # related clues
        clues = db_list_clues(conn, case_id=case_id)
        related_clues = [c["text"] for c in clues if s["name"].lower() in c["text"].lower()]
        evidence_items = list_evidence(conn, s["id"], case_id)
        allegations = list_allegations(conn, s['id'], case_id)
        s["relatedClues"] = related_clues
        s["evidence"] = evidence_items
        s["allegations"] = allegations
        return jsonify(s)


@app.get('/api/suspects/<sid>/attribution')
def api_suspect_attribution(sid: str):
    case_id = request.args.get('case_id') or 'default'
    with get_conn() as conn:
        if not db_get_suspect(conn, sid):
            return jsonify({'error': 'suspect not found'}), 404
        rows = fetch_attributions(conn, sid, case_id)
        # group by clue_id
        grouped: dict[int, list[dict[str, Any]]] = {}
        for r in rows:
            grouped.setdefault(r['clue_id'], []).append({'token': r['token'], 'weight': r['weight']})
        # fetch clue texts for context
        clue_map = {c['id']: c['text'] for c in db_list_clues(conn, case_id=case_id)}
    out = []
    for cid, toks in grouped.items():
        out.append({
            'clue_id': cid,
            'clue_text': clue_map.get(cid, ''),
            'tokens': toks
        })
    return jsonify({'suspect_id': sid, 'case_id': case_id, 'attribution': out})


@app.get('/api/suspects/<sid>/explain')
def api_explain_suspect(sid: str):
    """Return top contributing tokens for a suspect label from the linear model.
    This is a heuristic explanation: shows n-grams with highest positive coefficient.
    """
    try:
        if not Path(MODEL_PATH).exists():
            train_and_save(BASE_DIR / "inputs" / "sample_training.json")
        model = rank_labels.__globals__['load_model']()  # reuse existing load function without circular import
        pipe = model
        clf = pipe.named_steps['clf']
        vec = pipe.named_steps['tfidf']
        classes = list(clf.classes_)
        target_label = sid
        if target_label not in classes:
            # attempt case-insensitive match
            lowered = {c.lower(): c for c in classes}
            if target_label.lower() in lowered:
                target_label = lowered[target_label.lower()]
            else:
                return jsonify({"error": "Label not in model"}), 404
        import numpy as np
        idx = classes.index(target_label)
        coefs = clf.coef_[idx]
        feature_names = vec.get_feature_names_out()
        order = np.argsort(coefs)[::-1]
        top_n = []
        for i in order[:20]:
            w = float(coefs[i])
            if w <= 0:
                break
            top_n.append({"token": feature_names[i], "weight": w})
        return jsonify({
            "label": target_label,
            "top_tokens": top_n,
            "total_positive": len([c for c in coefs if c > 0])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/suspects")
@auth_required
def api_create_suspect():
    data = request.get_json(force=True)
    required = ["id", "name"]
    if not all(k in data for k in required):
        return jsonify({"error": "id and name required"}), 400
    with get_conn() as conn:
        if db_get_suspect(conn, data["id"]):
            return jsonify({"error": "id exists"}), 409
        insert_suspect(conn, data["id"], data["name"], data.get("bio", ''), data.get("avatar", ''), data.get("status", 'unknown'), data.get("tags"), data.get('case_id','default'))
        return jsonify({"ok": True}), 201


@app.patch("/api/suspects/<sid>")
@auth_required
def api_update_suspect(sid: str):
    data = request.get_json(force=True) or {}
    with get_conn() as conn:
        if not db_get_suspect(conn, sid):
            return jsonify({"error": "Not found"}), 404
        update_suspect(conn, sid, **data)
        return jsonify({"ok": True})


@app.delete("/api/suspects/<sid>")
@auth_required
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
    case_id = request.args.get('case_id') or 'default'
    hide_duplicates = request.args.get('hide_duplicates') == '1'
    min_quality = request.args.get('min_quality')
    annotation_label = request.args.get('annotation_label')
    try:
        min_q_val = float(min_quality) if min_quality is not None else None
    except ValueError:
        min_q_val = None
    with get_conn() as conn:
        rows = db_list_clues(conn, suspect_id, case_id, hide_duplicates=hide_duplicates, min_quality=min_q_val, annotation_label=annotation_label)
        if limit_param:
            try:
                lim = int(limit_param)
                rows = rows[:lim]
            except ValueError:
                pass
        return jsonify(rows)


@app.post("/api/clues")
@auth_required
def api_add_clue():
    data = request.get_json(force=True)
    text = data.get("text")
    if not text:
        return jsonify({"error": "text required"}), 400
    source_type = data.get('source_type') or 'manual'
    with get_conn() as conn:
        insert_clue(conn, text, data.get("suspect_id"), data.get('case_id','default'), source_type=source_type)
        return jsonify({"ok": True}), 201


@app.delete("/api/clues/<int:clue_id>")
@auth_required
def api_delete_clue(clue_id: int):
    with get_conn() as conn:
        delete_clue(conn, clue_id)
        return jsonify({"ok": True})


@app.post('/api/clues/<int:clue_id>/annotate')
@auth_required
def api_annotate_clue(clue_id: int):
    data = request.get_json(force=True) or {}
    label = data.get('label')
    notes = data.get('notes')
    if not label:
        return jsonify({'error': 'label required'}), 400
    try:
        with get_conn() as conn:
            annotate_clue(conn, clue_id, label, notes)
        return jsonify({'ok': True})
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400


@app.post('/api/clues/recompute_duplicates')
@auth_required
def api_recompute_duplicates():
    data = request.get_json(silent=True) or {}
    case_id = data.get('case_id') or request.args.get('case_id') or 'default'
    threshold = float(data.get('threshold', 0.85))
    with get_conn() as conn:
        try:
            recompute_duplicates(conn, case_id, threshold=threshold)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'ok': True, 'case_id': case_id, 'threshold': threshold})


@app.post('/api/clues/recompute_quality')
@auth_required
def api_recompute_quality():
    data = request.get_json(silent=True) or {}
    case_id = data.get('case_id') or request.args.get('case_id')
    with get_conn() as conn:
        try:
            recompute_clue_quality(conn, case_id)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'ok': True, 'case_id': case_id or 'ALL'})


@app.get("/api/evidence/<sid>")
def api_list_evidence(sid: str):
    case_id = request.args.get('case_id') or 'default'
    with get_conn() as conn:
        return jsonify(list_evidence(conn, sid, case_id))


@app.post("/api/evidence/<sid>")
@auth_required
def api_add_evidence(sid: str):
    data = request.get_json(force=True)
    with get_conn() as conn:
        if not db_get_suspect(conn, sid):
            return jsonify({"error": "Suspect not found"}), 404
    insert_evidence(conn, sid, data.get("type", "misc"), data.get("summary", ""), float(data.get("weight", 0)), data.get('case_id','default'))
    return jsonify({"ok": True, "evidence": list_evidence(conn, sid, data.get('case_id','default'))}), 201


@app.get('/api/suspects/<sid>/allegations')
def api_list_allegations(sid: str):
    case_id = request.args.get('case_id') or 'default'
    with get_conn() as conn:
        if not db_get_suspect(conn, sid):
            return jsonify({'error': 'Suspect not found'}), 404
        return jsonify(list_allegations(conn, sid, case_id))


@app.post('/api/suspects/<sid>/allegations')
@auth_required
def api_add_allegation(sid: str):
    data = request.get_json(force=True) or {}
    offense = data.get('offense')
    if not offense:
        return jsonify({'error': 'offense required'}), 400
    severity = data.get('severity', 'medium')
    if severity not in {'low','medium','high'}:
        severity = 'medium'
    with get_conn() as conn:
        if not db_get_suspect(conn, sid):
            return jsonify({'error': 'Suspect not found'}), 404
        insert_allegation(conn, sid, offense, data.get('description',''), severity, data.get('case_id','default'))
        return jsonify({'ok': True}), 201


@app.delete('/api/allegations/<int:aid>')
@auth_required
def api_delete_allegation(aid: int):
    with get_conn() as conn:
        delete_allegation(conn, aid)
    return jsonify({'ok': True})


@app.patch("/api/evidence/<int:evidence_id>")
@auth_required
def api_update_evidence(evidence_id: int):
    data = request.get_json(force=True) or {}
    with get_conn() as conn:
        update_evidence(conn, evidence_id, **data)
        return jsonify({"ok": True})


@app.delete("/api/evidence/<int:evidence_id>")
@auth_required
def api_delete_evidence(evidence_id: int):
    with get_conn() as conn:
        delete_evidence(conn, evidence_id)
        return jsonify({"ok": True})


@app.post("/api/rescore")
@auth_required
def api_rescore():
    # Force recomputation of scores and persist timestamp
    case_id = request.args.get('case_id') or 'default'
    with get_conn() as conn:
        suspects = db_list_suspects(conn, case_id)
        try:
            text = aggregate_clues_text(conn, case_id)
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


@app.get('/api/search')
def api_search():
    """Semantic (or TF-IDF fallback) search over clues.

    Query Params:
      q: search string (required)
      k: top k (default 5)
      case_id: case scope (default 'default')
      refresh: if '1' force rebuild index
    """
    q = request.args.get('q')
    if not q:
        return jsonify({"error": "q required"}), 400
    try:
        k = int(request.args.get('k', '5'))
    except ValueError:
        k = 5
    case_id = request.args.get('case_id') or 'default'
    if request.args.get('refresh') == '1':
        semantic_refresh(case_id)
    results = semantic_search(q, k=k, case_id=case_id)
    return jsonify({
        "query": q,
        "k": k,
        "backend": semantic_backend(),
        "case_id": case_id,
        "results": results
    })


@app.post('/api/analysis')
def api_case_analysis():
    data = request.get_json(silent=True) or {}
    case_id = data.get('case_id') or request.args.get('case_id') or 'default'
    style = data.get('style', 'brief')
    with get_conn() as conn:
        if case_id and not get_case(conn, case_id):
            return jsonify({'error': 'case not found'}), 404
        suspects = db_list_suspects(conn, case_id)
        # Reuse clues listing (not aggregating into one string so we can selectively feed)
        clues_rows = db_list_clues(conn, case_id=case_id)
        clues = [r['text'] for r in clues_rows]
    report = generate_case_analysis(case_id, clues, suspects, style=style)
    return jsonify(report)


@app.post('/api/documents/upload')
@auth_required
def api_upload_document():
    """Upload a PDF, extract text, store it, and attempt suspect inference.

    Multipart form fields:
      file: PDF file (required)
      case_id: optional case id (default 'default')
      auto_clues: '1' to also create clue entries (splitting by lines)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'file required'}), 400
    pdf = request.files['file']
    if not pdf.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'only pdf accepted'}), 400
    case_id = request.form.get('case_id') or 'default'
    auto_clues = request.form.get('auto_clues') == '1'
    try:
        import pdfplumber  # type: ignore
    except Exception:
        return jsonify({'error': 'pdfplumber not installed'}), 500
    from werkzeug.utils import secure_filename
    safe_name = secure_filename(pdf.filename)
    tmp_path = BASE_DIR / 'uploads'
    tmp_path.mkdir(exist_ok=True)
    local_path = tmp_path / safe_name
    pdf.save(local_path)
    # Extract text
    try:
        text_parts = []
        with pdfplumber.open(str(local_path)) as doc:
            for page in doc.pages:
                t = page.extract_text() or ''
                if t.strip():
                    text_parts.append(t.strip())
        full_text = '\n'.join(text_parts)
    except Exception as e:
        return jsonify({'error': f'failed to parse pdf: {e}'}), 500
    # Persist document
    with get_conn() as conn:
        doc_id = insert_document(conn, filename=str(local_path.name), original_name=pdf.filename, text=full_text, case_id=case_id)
        # Chunking for RAG: simple paragraph / line chunks ~400 chars
        chunk_size = 500
        chunks = []
        buf = ''
        for line in full_text.splitlines():
            if len(buf) + len(line) + 1 <= chunk_size:
                buf += ('\n' if buf else '') + line
            else:
                if buf:
                    chunks.append(buf.strip())
                buf = line
        if buf:
            chunks.append(buf.strip())
        # Optional embedding (reuse sentence transformer if available)
        embed_vectors = []
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            embed_vectors = _emb_model.encode(chunks, normalize_embeddings=True).tolist()  # type: ignore
        except Exception:
            embed_vectors = []
        for i, ch in enumerate(chunks):
            emb = embed_vectors[i] if i < len(embed_vectors) else None
            try:
                insert_document_chunk(conn, doc_id, case_id, i, ch, emb)
            except Exception:
                pass
        # Optionally create clues for each non-empty line (cap to 100 to avoid noise)
        created_clues = 0
        if auto_clues:
            for line in full_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                insert_clue(conn, line[:500], None, case_id)  # no initial suspect mapping
                created_clues += 1
                if created_clues >= 100:
                    break
        # After ingest, attempt model-driven suspect ranking on the document text
        try:
            if not Path(MODEL_PATH).exists():
                train_and_save(BASE_DIR / 'inputs' / 'sample_training.json')
            ranked = rank_labels([full_text], top_k=5)
        except Exception:
            ranked = []
    return jsonify({
        'ok': True,
        'document_id': doc_id,
        'filename': safe_name,
        'chars': len(full_text),
        'suspect_suggestions': [{'label': l, 'score': s} for l, s in ranked]
    }), 201


@app.get('/api/documents')
def api_list_documents():
    case_id = request.args.get('case_id') or 'default'
    with get_conn() as conn:
        docs = list_documents(conn, case_id)
    # Return without full text for listing
    for d in docs:
        d.pop('text', None)
    return jsonify(docs)


@app.post('/api/feedback')
@auth_required
def api_add_feedback():
    data = request.get_json(force=True) or {}
    suspect_id = data.get('suspect_id')
    decision = data.get('decision')
    if not suspect_id or not decision:
        return jsonify({'error': 'suspect_id and decision required'}), 400
    case_id = data.get('case_id') or 'default'
    with get_conn() as conn:
        s = db_get_suspect(conn, suspect_id)
        if not s:
            return jsonify({'error': 'suspect not found'}), 404
        try:
            fid = insert_feedback(
                conn,
                suspect_id=suspect_id,
                decision=decision,
                rank_at_feedback=data.get('rank_at_feedback'),
                composite_score=data.get('composite_score'),
                ml_score=data.get('ml_score'),
                evidence_score=data.get('evidence_score'),
                offense_boost=data.get('offense_boost'),
                case_id=case_id,
                clue_id=data.get('clue_id')
            )
        except ValueError as ve:
            return jsonify({'error': str(ve)}), 400
    return jsonify({'ok': True, 'id': fid}), 201


@app.get('/api/feedback')
def api_list_feedback():
    case_id = request.args.get('case_id')
    limit = request.args.get('limit', '100')
    try:
        lim = int(limit)
    except ValueError:
        lim = 100
    with get_conn() as conn:
        rows = list_feedback(conn, case_id, lim)
    return jsonify(rows)


@app.get('/api/feedback/stats')
def api_feedback_stats():
    case_id = request.args.get('case_id')
    with get_conn() as conn:
        stats = feedback_stats(conn, case_id)
    return jsonify(stats)


@app.get('/api/qa')
def api_qa():
    """Hybrid RAG QA endpoint.

    Query params:
      q: question text (required)
      k: top chunks (default 5)
      case_id: scope
    Strategy:
      1. Embed question -> cosine over stored document_chunks embeddings (if exist)
      2. Combine with top semantic clue search results
      3. Compose concise answer heuristic (future: LLM) returning citations.
    """
    q = request.args.get('q')
    if not q:
        return jsonify({'error': 'q required'}), 400
    try:
        k = int(request.args.get('k','5'))
    except ValueError:
        k = 5
    case_id = request.args.get('case_id') or 'default'
    with get_conn() as conn:
        chunks = list_chunks(conn, case_id, limit=5000)
    ranked_chunks = []
    import math
    if chunks and any(c.get('embedding') for c in chunks):
        # embed question
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _model_q = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            q_vec = _model_q.encode([q], normalize_embeddings=True)[0]  # type: ignore
            # cosine
            def cos(a,b):
                import numpy as np
                a=np.array(a); b=np.array(b)
                return float((a@b)/( (a.dot(a)**0.5)*(b.dot(b)**0.5)+1e-9 ))
            scored = []
            for c in chunks:
                emb = c.get('embedding')
                if not emb:
                    continue
                try:
                    score = cos(q_vec, emb)
                except Exception:
                    score = 0.0
                scored.append((score, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            ranked_chunks = [ {'chunk_id': sc[1]['id'], 'text': sc[1]['text'], 'score': round(sc[0],4)} for sc in scored[:k] ]
        except Exception:
            ranked_chunks = []
    # Clue results
    clue_hits = semantic_search(q, k=k, case_id=case_id)
    # Heuristic answer: concatenate top lines containing overlapping keywords
    key_terms = [w for w in q.split() if len(w) > 3]
    answer_parts = []
    for ch in ranked_chunks:
        if any(t.lower() in ch['text'].lower() for t in key_terms):
            snippet = ch['text'][:180].replace('\n',' ')
            answer_parts.append(snippet + ('...' if len(ch['text'])>180 else ''))
            if len(answer_parts) >= 3:
                break
    if not answer_parts:
        for c in clue_hits:
            if any(t.lower() in c['text'].lower() for t in key_terms):
                answer_parts.append(c['text'][:160] + ('...' if len(c['text'])>160 else ''))
                if len(answer_parts) >= 3:
                    break
    answer = ' '.join(answer_parts) if answer_parts else 'Insufficient context to answer definitively.'
    return jsonify({
        'question': q,
        'answer': answer,
        'chunks': ranked_chunks,
        'clues': clue_hits,
        'case_id': case_id
    })


@app.get('/api/timeline')
def api_timeline():
    """Return (and if empty, attempt extraction) of timeline events for a case."""
    case_id = request.args.get('case_id') or 'default'
    with get_conn() as conn:
        existing = list_events(conn, case_id)
        if existing:
            return jsonify({'case_id': case_id, 'events': existing})
    # Extract basic dates from clues
    import re, datetime
    date_pattern = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", re.IGNORECASE)
    with get_conn() as conn:
        clues_rows = db_list_clues(conn, case_id=case_id)
        for c in clues_rows:
            matches = date_pattern.findall(c['text'])
            for m in matches:
                norm = None
                try:
                    if re.match(r"\d{4}-\d{2}-\d{2}", m):
                        norm = m + 'T00:00:00'
                    elif re.match(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", m):
                        parts = re.split(r"[/-]", m)
                        if len(parts[-1]) == 2:
                            parts[-1] = '20'+parts[-1]
                        mm, dd, yyyy = parts[0], parts[1], parts[2]
                        norm = f"{yyyy.zfill(4)}-{mm.zfill(2)}-{dd.zfill(2)}T00:00:00"
                except Exception:
                    norm = None
                insert_event(conn, case_id, 'clue', c['id'], c['text'][:160], m, norm, None)
        events = list_events(conn, case_id)
    return jsonify({'case_id': case_id, 'events': events})


@app.get('/api/graph')
def api_graph():
    case_id = request.args.get('case_id') or 'default'
    graph = build_graph(case_id)
    return jsonify(graph)


@app.post('/api/simulate')
@auth_required
def api_simulate():
    """Ephemeral composite scoring simulation.

    Payload JSON fields:
      case_id: scope (default 'default')
      overrides: {
          evidence_weights: {suspect_id: new_total_weight},
          remove_offenses: {suspect_id: [offense substrings to remove]},
          alpha: float (optional override),
          offense_beta: float (optional override)
      }
    Returns: new composite scores + deltas vs current persisted.
    """
    data = request.get_json(force=True) or {}
    case_id = data.get('case_id') or 'default'
    overrides = data.get('overrides', {}) or {}
    alpha = float(overrides.get('alpha', 0.7))
    alpha = max(0.0, min(1.0, alpha))
    offense_beta = float(overrides.get('offense_beta', 0.1))
    offense_beta = max(0.0, min(0.5, offense_beta))
    ev_over = overrides.get('evidence_weights', {}) or {}
    remove_off = overrides.get('remove_offenses', {}) or {}
    severity_value_map = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
    with get_conn() as conn:
        suspects = db_list_suspects(conn, case_id)
        original = {s['id'].lower(): s.get('composite_score') for s in suspects}
        # recompute evidence totals (overridden if provided)
        evidence_totals: dict[str, float] = {}
        for s in suspects:
            sid = s['id'].lower()
            evid_items = list_evidence(conn, s['id'], case_id)
            tot = sum((e.get('weight') or 0.0) for e in evid_items)
            evidence_totals[sid] = float(ev_over.get(sid, tot))
        # ML scores reused from current endpoint (simple: use persisted last_score or recompute quick)
        # Use last_score (if available) to avoid heavy recompute
        ml_scores = {}
        for s in suspects:
            ml_scores[s['id'].lower()] = s.get('score') or s.get('last_score') or 0.0
        results = []
        for s in suspects:
            sid = s['id'].lower()
            ml = ml_scores.get(sid, 0.0)
            ev_norm = min(1.0, evidence_totals.get(sid, 0.0) / 5.0)
            # Adjust offenses: remove if substring match requested
            allegations = list_allegations(conn, s['id'], case_id)
            removals = [r.lower() for r in remove_off.get(sid, [])]
            kept = []
            for a in allegations:
                if any(rem in a['offense'].lower() for rem in removals):
                    continue
                kept.append(a)
            if kept:
                severity_sorted = sorted(kept, key=lambda a: {'high':3,'medium':2,'low':1}.get(a.get('severity','low'),1), reverse=True)
                sev = severity_value_map.get(severity_sorted[0].get('severity','low'), 0.0)
            else:
                sev = 0.0
            offense_boost = offense_beta * sev
            base_comp = alpha * ml + (1-alpha) * ev_norm
            comp = min(1.0, base_comp + offense_boost)
            results.append({
                'suspect_id': s['id'],
                'ml_score': ml,
                'evidence_score': ev_norm,
                'offense_boost': offense_boost,
                'composite_base': base_comp,
                'composite_score': comp,
                'delta': None if original.get(sid) is None else comp - (original[sid] or 0.0)
            })
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    return jsonify({
        'case_id': case_id,
        'alpha': alpha,
        'offense_beta': offense_beta,
        'simulated': results
    })


@app.get('/api/suspects/<sid>/risk_explanation')
def api_risk_explanation(sid: str):
    """Return structured factors and a narrative explaining risk components."""
    case_id = request.args.get('case_id') or 'default'
    with get_conn() as conn:
        s = db_get_suspect(conn, sid)
        if not s:
            return jsonify({'error': 'suspect not found'}), 404
        # gather evidence + allegations + feedback summary
        evidence_items = list_evidence(conn, sid, case_id)
        allegations = list_allegations(conn, sid, case_id)
        fb = [f for f in list_feedback(conn, case_id, 500) if f['suspect_id']==sid]
    ev_total = sum((e.get('weight') or 0.0) for e in evidence_items)
    ev_norm = min(1.0, ev_total / 5.0)
    severity_value_map = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
    top_sev = 0.0
    primary_offense = None
    if allegations:
        ordered = sorted(allegations, key=lambda a: {'high':3,'medium':2,'low':1}.get(a.get('severity','low'),1), reverse=True)
        primary_offense = ordered[0]['offense']
        top_sev = severity_value_map.get(ordered[0].get('severity','low'), 0.0)
    confirmations = sum(1 for f in fb if f['decision']=='confirm')
    rejections = sum(1 for f in fb if f['decision']=='reject')
    factors = {
        'evidence_density': ev_norm,
        'primary_offense': primary_offense,
        'offense_severity_value': top_sev,
        'feedback_confirmations': confirmations,
        'feedback_rejections': rejections,
        'composite_score': s.get('composite_score')
    }
    # Narrative template (no external LLM call to keep deterministic)
    parts = []
    score = s.get('composite_score') or 0.0
    if ev_norm >= 0.6:
        parts.append(f"Strong evidence accumulation ({ev_norm:.2f} normalized).")
    elif ev_norm >= 0.3:
        parts.append(f"Moderate evidence presence ({ev_norm:.2f}).")
    else:
        parts.append(f"Sparse direct evidence ({ev_norm:.2f}).")
    if primary_offense:
        sev_desc = 'high-severity' if top_sev >= 0.9 else 'notable' if top_sev >= 0.6 else 'lower-severity'
        parts.append(f"Primary offense '{primary_offense}' is {sev_desc} (sev={top_sev:.2f}).")
    if confirmations and confirmations > rejections:
        parts.append(f"Analyst feedback leans confirm ({confirmations} confirm vs {rejections} reject).")
    elif rejections > confirmations:
        parts.append(f"Analyst skepticism noted ({rejections} reject vs {confirmations} confirm).")
    else:
        parts.append("Limited analyst feedback so far.")
    if score >= 0.6:
        parts.append(f"Overall composite score {score:.2f} places this suspect in HIGH risk band.")
    elif score >= 0.4:
        parts.append(f"Composite score {score:.2f} indicates MEDIUM risk requiring monitoring.")
    else:
        parts.append(f"Composite score {score:.2f} remains LOW; watch for new evidence.")
    narrative = ' '.join(parts)
    return jsonify({'suspect_id': sid, 'case_id': case_id, 'factors': factors, 'narrative': narrative})


@app.get('/api/metrics')
def api_metrics():
    """Return high-level system & model metrics for dashboard.

    Includes:
      - Entity counts
      - Feedback aggregate stats
      - Score distribution quantiles
      - Severity distribution
      - Evidence weight statistics
      - Document ingestion stats
      - Model backend & config defaults
    """
    case_id = request.args.get('case_id')  # if provided scope counts to case where applicable
    with get_conn() as conn:
        cur = conn.cursor()
        counts = {}
        # Count helpers (case-scoped where meaningful)
        def _count(table: str) -> int:
            if table in {'suspects','clues','evidence','allegations','documents'} and case_id:
                cur.execute(f"SELECT COUNT(*) as c FROM {table} WHERE case_id=?", (case_id,))
            else:
                cur.execute(f"SELECT COUNT(*) as c FROM {table}")
            return int(cur.fetchone()['c'])
        counts['suspects'] = _count('suspects')
        counts['clues'] = _count('clues')
        counts['evidence'] = _count('evidence')
        counts['allegations'] = _count('allegations')
        counts['documents'] = _count('documents') if 'documents' in counts else _count('documents')
        counts['cases'] = _count('cases')
        # Score distribution (composite)
        if case_id:
            cur.execute("SELECT composite_score FROM suspects WHERE case_id=? AND composite_score IS NOT NULL", (case_id,))
        else:
            cur.execute("SELECT composite_score FROM suspects WHERE composite_score IS NOT NULL")
        scores = [row['composite_score'] for row in cur.fetchall() if row['composite_score'] is not None]
        def _quantiles(vals: list[float]):
            if not vals:
                return {}
            import numpy as np
            arr = np.array(vals)
            return {
                'min': float(arr.min()),
                'p25': float(np.percentile(arr, 25)),
                'p50': float(np.percentile(arr, 50)),
                'p75': float(np.percentile(arr, 75)),
                'max': float(arr.max()),
                'count': int(arr.size)
            }
        score_dist = _quantiles(scores)
        # Severity distribution
        if case_id:
            cur.execute("SELECT severity, COUNT(*) as c FROM allegations WHERE case_id=? GROUP BY severity", (case_id,))
        else:
            cur.execute("SELECT severity, COUNT(*) as c FROM allegations GROUP BY severity")
        sev_rows = cur.fetchall()
        severity = {r['severity']: r['c'] for r in sev_rows}
        # Evidence stats
        if case_id:
            cur.execute("SELECT AVG(weight) as avg_w, SUM(weight) as sum_w FROM evidence WHERE case_id=?", (case_id,))
        else:
            cur.execute("SELECT AVG(weight) as avg_w, SUM(weight) as sum_w FROM evidence")
        ev_row = cur.fetchone()
        evidence_stats = {
            'avg_weight': float(ev_row['avg_w']) if ev_row and ev_row['avg_w'] is not None else None,
            'total_weight': float(ev_row['sum_w']) if ev_row and ev_row['sum_w'] is not None else 0.0,
        }
        # Document ingestion stats
        if case_id:
            cur.execute("SELECT COUNT(*) as c, SUM(LENGTH(text)) as total_chars, AVG(LENGTH(text)) as avg_chars FROM documents WHERE case_id=?", (case_id,))
        else:
            cur.execute("SELECT COUNT(*) as c, SUM(LENGTH(text)) as total_chars, AVG(LENGTH(text)) as avg_chars FROM documents")
        drow = cur.fetchone()
        ingestion = {
            'documents': int(drow['c']) if drow and drow['c'] is not None else 0,
            'total_chars': int(drow['total_chars']) if drow and drow['total_chars'] is not None else 0,
            'avg_chars': float(drow['avg_chars']) if drow and drow['avg_chars'] is not None else None,
        }
        # Last scored timestamp
        if case_id:
            cur.execute("SELECT MAX(last_scored_at) as latest FROM suspects WHERE case_id=?", (case_id,))
        else:
            cur.execute("SELECT MAX(last_scored_at) as latest FROM suspects")
        ls = cur.fetchone()['latest']
        # Feedback stats (reuse function)
        fb = feedback_stats(conn, case_id)
    # Model backend detection (cheap load attempt)
    try:
        from ml_transformer import load_transformer  # type: ignore
        model, _, _ = load_transformer()
        backend = 'transformer' if model else 'logreg'
    except Exception:
        backend = 'logreg'
    return jsonify({
        'counts': counts,
        'scores': score_dist,
        'severity': severity,
        'evidence': evidence_stats,
        'ingestion': ingestion,
        'feedback': fb,
        'model': {
            'backend': backend,
            'alpha_default': 0.7,
            'offense_beta_default': 0.1
        },
        'data_freshness': ls,
        'case_id': case_id or 'ALL'
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
