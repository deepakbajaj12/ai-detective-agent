from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

from flask import Flask, jsonify, request, Response, stream_with_context, send_file
from flask_cors import CORS

try:  # package import (preferred when running `python -m src.api`)
    from src.utils import read_clues
    from src.ml_suspect_model import MODEL_PATH, train_and_save, rank_labels
    from src.ml_transformer import ensure_transformer_model, predict_labels as transformer_rank
except ImportError:  # fallback if executed as `python src/api.py`
    from utils import read_clues  # type: ignore
    from ml_suspect_model import MODEL_PATH, train_and_save, rank_labels  # type: ignore
    from ml_transformer import ensure_transformer_model, predict_labels as transformer_rank  # type: ignore
try:
    # optional: only needed for attention-based attribution
    try:
        from src.ml_transformer import load_transformer  # type: ignore
    except Exception:
        from ml_transformer import load_transformer  # type: ignore
except Exception:  # pragma: no cover
    load_transformer = None  # type: ignore
try:
    from src.semantic_search import search as semantic_search, refresh as semantic_refresh, backend_mode as semantic_backend, get_embedding_metrics as semantic_embedding_metrics
    from src.db import init_db, get_conn, list_suspects as db_list_suspects, get_suspect as db_get_suspect, insert_suspect, update_suspect, delete_suspect, list_clues as db_list_clues, insert_clue, delete_clue, list_evidence, insert_evidence, update_evidence, delete_evidence, aggregate_clues_text, persist_scores, persist_composite_scores, list_cases, get_case, insert_case, list_allegations, insert_allegation, delete_allegation, insert_document, list_documents, get_document, insert_feedback, list_feedback, feedback_stats, clear_attributions, insert_attribution, fetch_attributions, insert_document_chunk, list_chunks, insert_event, list_events, create_user, find_user, create_token, get_user_by_token, annotate_clue, recompute_duplicates, recompute_clue_quality, insert_model_version, list_model_versions, get_model_version, set_model_role, clear_role, get_active_model, get_shadow_model, update_model_metrics, insert_snapshot, list_snapshots, get_snapshot
except ImportError:  # fallback
    from semantic_search import search as semantic_search, refresh as semantic_refresh, backend_mode as semantic_backend, get_embedding_metrics as semantic_embedding_metrics  # type: ignore
    from db import init_db, get_conn, list_suspects as db_list_suspects, get_suspect as db_get_suspect, insert_suspect, update_suspect, delete_suspect, list_clues as db_list_clues, insert_clue, delete_clue, list_evidence, insert_evidence, update_evidence, delete_evidence, aggregate_clues_text, persist_scores, persist_composite_scores, list_cases, get_case, insert_case, list_allegations, insert_allegation, delete_allegation, insert_document, list_documents, get_document, insert_feedback, list_feedback, feedback_stats, clear_attributions, insert_attribution, fetch_attributions, insert_document_chunk, list_chunks, insert_event, list_events, create_user, find_user, create_token, get_user_by_token, annotate_clue, recompute_duplicates, recompute_clue_quality, insert_model_version, list_model_versions, get_model_version, set_model_role, clear_role, get_active_model, get_shadow_model, update_model_metrics, insert_snapshot, list_snapshots, get_snapshot  # type: ignore
try:
    from src.graph_builder import build_graph, analyze_graph  # type: ignore
    from src.gen_ai import generate_case_analysis, answer_with_context, stream_answer  # type: ignore
    from src.pdf_generator import save_report as save_pdf_report  # type: ignore
    from src.rag import advanced_retrieve  # type: ignore
    from src.events_bus import publish_event, sse_stream_generator  # type: ignore
except Exception:
    from graph_builder import build_graph, analyze_graph  # type: ignore
    from gen_ai import generate_case_analysis, answer_with_context, stream_answer  # type: ignore
    from rag import advanced_retrieve  # type: ignore
    from events_bus import publish_event, sse_stream_generator  # type: ignore
try:
    from src.jobs_backend import start_job, get_job, list_jobs as jobs_list, cancel_job as jobs_cancel, task_transformer_train, task_index_refresh, task_embeddings_refresh  # type: ignore
    from src.jobs_backend import backend_mode as job_backend_mode  # type: ignore
except Exception:
    from jobs_backend import start_job, get_job, list_jobs as jobs_list, cancel_job as jobs_cancel, task_transformer_train, task_index_refresh, task_embeddings_refresh  # type: ignore
    try:
        from jobs_backend import backend_mode as job_backend_mode  # type: ignore
    except Exception:
        def job_backend_mode():  # type: ignore
            return 'memory'


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
            "GET /api/graph/analytics": "Graph + centrality, communities, anomalies",
            "GET /api/model/versions": "List registered models & roles",
            "POST /api/model/register": "Register a new model version",
            "POST /api/model/promote": "Promote version to active (demote old)",
            "POST /api/model/shadow": "Set version as shadow (A/B)",
            "POST /api/model/rollback": "Rollback active to prior version",
            "POST /api/model/infer_ab": "Run A/B inference (active vs shadow)",
            "GET /api/model/eval/logs": "List recent model A/B inference logs",
            "GET /api/model/eval/stats": "Aggregate model evaluation statistics",
            "POST /api/snapshots": "Create a ranking snapshot",
            "GET /api/snapshots": "List snapshots",
            "GET /api/snapshots/compare": "Compare two snapshots side-by-side",
            "GET /api/suspects/<id>/waterfall": "Score component breakdown for visualization",
            "POST /api/auth/register": "Create a new user (initial open mode)",
            "POST /api/auth/login": "Get an API token",
            "GET /api/metrics": "System + model overview metrics",
            "POST /api/jobs/transformer_train": "Start background transformer training (optional deps)",
            "POST /api/jobs/index_refresh": "Rebuild semantic index for a single case in background",
            "POST /api/jobs/embeddings_refresh": "Rebuild embeddings/indexes for ALL cases sequentially",
            "GET /api/system": "System info (job backend mode, counts)",
            "GET /api/jobs": "List recent jobs (durable if Redis/RQ enabled)",
            "GET /api/jobs/<id>": "Poll job status",
            "POST /api/jobs/<id>/cancel": "Cancel a running or queued job (RQ only)"
        }
    })


# ---- Jobs API (lightweight background tasks) ----

@app.post('/api/jobs/transformer_train')
def api_job_transformer_train():
    data = request.get_json(silent=True) or {}
    rel = data.get('training_json') or 'inputs/sample_training.json'
    path = (BASE_DIR / rel).resolve()
    job_id = start_job('transformer_train', task_transformer_train, str(path))
    return jsonify({'ok': True, 'job_id': job_id}), 202


@app.post('/api/jobs/index_refresh')
def api_job_index_refresh():
    data = request.get_json(silent=True) or {}
    case_id = data.get('case_id') or 'default'
    job_id = start_job('index_refresh', task_index_refresh, case_id)
    return jsonify({'ok': True, 'job_id': job_id}), 202


@app.post('/api/jobs/embeddings_refresh')
def api_job_embeddings_refresh():
    """Trigger a global embeddings/index refresh across all cases.

    Background job will iterate through each case and rebuild semantic search indexes,
    recording per-case durations in the embedding metrics registry.
    Returns 202 with a job id that can be polled at /api/jobs/<id>.
    """
    # memory backend: we pass explicit task; RQ backend: job_type mapping ignores target
    job_id = start_job('embeddings_refresh', task_embeddings_refresh)
    return jsonify({'ok': True, 'job_id': job_id}), 202


@app.get('/api/jobs/<job_id>')
def api_job_status(job_id: str):
    j = get_job(job_id)
    if not j:
        return jsonify({'error': 'job not found'}), 404
    return jsonify(j)
# (removed duplicate legacy job endpoints)


@app.get('/api/jobs')
def api_jobs_list():
    try:
        lim = int(request.args.get('limit','50'))
    except ValueError:
        lim = 50
    rows = jobs_list(lim)
    return jsonify({'jobs': rows, 'count': len(rows)})


@app.post('/api/jobs/<job_id>/cancel')
def api_job_cancel(job_id: str):
    ok = jobs_cancel(job_id)
    if not ok:
        return jsonify({'error': 'cancel not supported or job not found'}), 400
    return jsonify({'ok': True, 'job_id': job_id})


@app.get('/api/system')
def api_system():
    """Return small system snapshot including jobs backend mode and minimal counts."""
    try:
        jb = job_backend_mode()
    except Exception:
        jb = 'memory'
    counts = {}
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as c FROM suspects")
            counts['suspects'] = int(cur.fetchone()['c'])
            cur.execute("SELECT COUNT(*) as c FROM clues")
            counts['clues'] = int(cur.fetchone()['c'])
            cur.execute("SELECT COUNT(*) as c FROM documents")
            counts['documents'] = int(cur.fetchone()['c'])
    except Exception:
        counts = {}
    return jsonify({'job_backend': jb, 'counts': counts})



# ---- Auth Utilities ----
import hashlib, secrets, os, time
import jwt  # PyJWT
from functools import wraps

def _hash_password(pw: str) -> str:
    # Retain legacy SHA256 for existing users; passlib could be integrated later.
    return hashlib.sha256(pw.encode('utf-8')).hexdigest()

JWT_SECRET = os.environ.get('AI_DETECTIVE_JWT_SECRET') or secrets.token_hex(32)
JWT_ALG = 'HS256'
JWT_EXP_SECONDS = int(os.environ.get('AI_DETECTIVE_JWT_EXP', '86400'))  # 24h default

def _issue_jwt(user: dict) -> str:
    payload = {
        'sub': user.get('id'),
        'username': user.get('username'),
        'role': user.get('role'),
        'iat': int(time.time()),
        'exp': int(time.time()) + JWT_EXP_SECONDS
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def _decode_jwt(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except Exception:
        return None


def auth_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        header = request.headers.get('Authorization')
        token = None
        if header and header.lower().startswith('bearer '):
            token = header.split(None,1)[1]
        if not token:
            return jsonify({'error': 'auth required'}), 401
        # Try JWT first
        claims = _decode_jwt(token)
        if claims:
            user = {
                'id': claims.get('sub'),
                'username': claims.get('username'),
                'role': claims.get('role')
            }
            request.user = user  # type: ignore
            return fn(*args, **kwargs)
        # Fallback legacy token lookup
        with get_conn() as conn:
            user = get_user_by_token(conn, token)
        if not user:
            return jsonify({'error': 'invalid token'}), 401
        request.user = user  # type: ignore
        return fn(*args, **kwargs)
    return wrapper

def role_required(*roles: str):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = getattr(request, 'user', None)
            if not user or user.get('role') not in roles:
                return jsonify({'error': 'forbidden'}), 403
            return fn(*args, **kwargs)
        return wrapper
    return deco


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
        # Issue JWT (legacy token creation skipped, but can keep compatibility if needed)
        user = {'id': uid, 'username': username, 'role': 'user'}
        jw = _issue_jwt(user)
    return jsonify({'ok': True, 'jwt': jw})


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
        jw = _issue_jwt(user)
    return jsonify({'ok': True, 'jwt': jw})


@app.get('/api/auth/me')
@auth_required
def api_auth_me():
    """Return basic user info for the bearer token (id, username, role)."""
    try:
        user = getattr(request, 'user', None)
        if not user:
            return jsonify({'error': 'no user context'}), 401
        return jsonify({'ok': True, 'user': {
            'id': user.get('id'),
            'username': user.get('username'),
            'role': user.get('role')
        }})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
        import torch  # type: ignore  # local import to avoid mandatory dependency if transformer unused
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
    try:
        publish_event('clue_added', {'text': text[:160], 'suspect_id': data.get('suspect_id'), 'case_id': data.get('case_id','default')})
    except Exception:
        pass
    return jsonify({"ok": True}), 201


@app.delete("/api/clues/<int:clue_id>")
@auth_required
def api_delete_clue(clue_id: int):
    with get_conn() as conn:
        delete_clue(conn, clue_id)
        return jsonify({"ok": True})


@app.get('/api/clues/<int:clue_id>/duplicates')
def api_list_clue_duplicates(clue_id: int):
    """List duplicates referencing this clue (duplicate_of_id = clue_id)."""
    case_id = request.args.get('case_id')  # optional narrowing
    with get_conn() as conn:
        # Confirm clue exists
        cur = conn.cursor()
        if case_id:
            cur.execute("SELECT id FROM clues WHERE id=? AND case_id=?", (clue_id, case_id))
        else:
            cur.execute("SELECT id FROM clues WHERE id=?", (clue_id,))
        if not cur.fetchone():
            return jsonify({'error': 'clue not found'}), 404
        if case_id:
            cur.execute("SELECT * FROM clues WHERE duplicate_of_id=? AND case_id=? ORDER BY id ASC", (clue_id, case_id))
        else:
            cur.execute("SELECT * FROM clues WHERE duplicate_of_id=? ORDER BY id ASC", (clue_id,))
        rows = [dict(r) for r in cur.fetchall()]
    return jsonify({'clue_id': clue_id, 'duplicates': rows, 'count': len(rows)})


# ---- Model Version Registry & A/B Endpoints ----
@app.get('/api/model/versions')
def api_model_versions():
    with get_conn() as conn:
        return jsonify(list_model_versions(conn))


@app.post('/api/model/register')
@auth_required
def api_model_register():
    data = request.get_json(force=True) or {}
    version_tag = data.get('version_tag')
    model_type = data.get('model_type') or 'logreg'
    path = data.get('path')
    metrics = data.get('metrics') or {}
    if not version_tag:
        return jsonify({'error': 'version_tag required'}), 400
    with get_conn() as conn:
        try:
            insert_model_version(conn, version_tag, model_type, path, 'archived', metrics)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'version_tag': version_tag})


@app.post('/api/model/promote')
@auth_required
def api_model_promote():
    data = request.get_json(force=True) or {}
    version_tag = data.get('version_tag')
    if not version_tag:
        return jsonify({'error': 'version_tag required'}), 400
    with get_conn() as conn:
        mv = get_model_version(conn, version_tag)
        if not mv:
            return jsonify({'error': 'version not found'}), 404
        clear_role(conn, 'active')
        set_model_role(conn, version_tag, 'active')
    return jsonify({'ok': True, 'active': version_tag})


@app.post('/api/model/shadow')
@auth_required
def api_model_shadow():
    data = request.get_json(force=True) or {}
    version_tag = data.get('version_tag')
    if not version_tag:
        return jsonify({'error': 'version_tag required'}), 400
    with get_conn() as conn:
        mv = get_model_version(conn, version_tag)
        if not mv:
            return jsonify({'error': 'version not found'}), 404
        clear_role(conn, 'shadow')
        set_model_role(conn, version_tag, 'shadow')
    return jsonify({'ok': True, 'shadow': version_tag})


@app.post('/api/model/rollback')
@auth_required
def api_model_rollback():
    """Rollback: set the most recent archived (excluding current active) as active.
    Simple heuristic: pick latest archived row.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        # fetch current active
        cur.execute("SELECT version_tag FROM model_versions WHERE role='active' ORDER BY created_at DESC LIMIT 1")
        active = cur.fetchone()
        active_tag = active['version_tag'] if active else None
        # find candidate archived (exclude active)
        if active_tag:
            cur.execute("SELECT version_tag FROM model_versions WHERE role='archived' AND version_tag<>? ORDER BY created_at DESC LIMIT 1", (active_tag,))
        else:
            cur.execute("SELECT version_tag FROM model_versions WHERE role='archived' ORDER BY created_at DESC LIMIT 1")
        cand = cur.fetchone()
        if not cand:
            return jsonify({'error': 'no archived version available'}), 400
        clear_role(conn, 'active')
        set_model_role(conn, cand['version_tag'], 'active')
    return jsonify({'ok': True, 'active': cand['version_tag'], 'previous': active_tag})


@app.post('/api/model/infer_ab')
def api_model_infer_ab():
    data = request.get_json(silent=True) or {}
    text = data.get('text')
    case_id = data.get('case_id') or 'default'
    if not text:
        # fallback: aggregate clues
        with get_conn() as conn:
            text = aggregate_clues_text(conn, case_id)
    # active inference uses existing pipeline logic; shadow is illustrative
    import time
    t0 = time.time()
    try:
        if not Path(MODEL_PATH).exists():
            train_and_save(BASE_DIR / 'inputs' / 'sample_training.json')
        active_ranked = rank_labels([text], top_k=5)
    except Exception as e:
        return jsonify({'error': f'active inference failed: {e}'}), 500
    # Shadow attempt: if transformer available, or reuse active with noise
    shadow_ranked = []
    with get_conn() as conn:
        shadow_meta = get_shadow_model(conn)
    try:
        # prefer transformer for shadow if not already active backend
        try:
            ensure_transformer_model(BASE_DIR / 'inputs' / 'sample_training.json')
            shadow_ranked = transformer_rank([text], top_k=5)
        except Exception:
            # fallback: perturb active scores slightly
            shadow_ranked = [(l, max(0.0, min(1.0, s * 0.97))) for l, s in active_ranked]
    except Exception:
        shadow_ranked = []
    latency_ms = (time.time() - t0) * 1000.0
    # Persist inference log
    try:
        with get_conn() as conn:
            from src.db import insert_inference_log  # type: ignore
            insert_inference_log(conn, input_chars=len(text), case_id=case_id, query_type='ab_compare', prompt=text,
                                 active_backend='logreg', shadow_backend=('transformer' if shadow_ranked else None),
                                 active_payload=[{'label': l, 'score': s} for l, s in active_ranked],
                                 shadow_payload=[{'label': l, 'score': s} for l, s in shadow_ranked], latency_ms=latency_ms)
    except Exception:
        pass
    return jsonify({
        'input_chars': len(text),
        'active': [{'label': l, 'score': s} for l, s in active_ranked],
        'shadow': [{'label': l, 'score': s} for l, s in shadow_ranked],
        'shadow_version': shadow_meta.get('version_tag') if shadow_meta else None,
        'latency_ms': latency_ms
    })


@app.get('/api/model/eval/logs')
def api_model_eval_logs():
    limit_param = request.args.get('limit','100')
    case_id = request.args.get('case_id')
    try:
        limit = int(limit_param)
    except ValueError:
        limit = 100
    from src.db import list_inference_logs  # type: ignore
    with get_conn() as conn:
        rows = list_inference_logs(conn, limit=limit, case_id=case_id)
    return jsonify({'logs': rows, 'count': len(rows)})


@app.get('/api/model/eval/stats')
def api_model_eval_stats():
    case_id = request.args.get('case_id')
    from src.db import inference_stats  # type: ignore
    with get_conn() as conn:
        stats = inference_stats(conn, case_id=case_id)
    return jsonify({'case_id': case_id or 'ALL', 'stats': stats})


# ---- Score Snapshots ----
@app.post('/api/snapshots')
@auth_required
def api_create_snapshot():
    data = request.get_json(force=True) or {}
    case_id = data.get('case_id') or 'default'
    label = data.get('label')
    with get_conn() as conn:
        suspects = db_list_suspects(conn, case_id)
        sid = insert_snapshot(conn, label, case_id, suspects)
    return jsonify({'ok': True, 'snapshot_id': sid})


@app.get('/api/snapshots')
def api_list_snapshots():
    case_id = request.args.get('case_id')
    limit = request.args.get('limit','50')
    try:
        lim = int(limit)
    except ValueError:
        lim = 50
    with get_conn() as conn:
        rows = list_snapshots(conn, case_id, lim)
    return jsonify(rows)


@app.get('/api/snapshots/compare')
def api_compare_snapshots():
    a = request.args.get('a')
    b = request.args.get('b')
    if not a or not b:
        return jsonify({'error': 'a and b snapshot ids required'}), 400
    try:
        aid = int(a); bid = int(b)
    except ValueError:
        return jsonify({'error': 'invalid snapshot ids'}), 400
    with get_conn() as conn:
        sa = get_snapshot(conn, aid)
        sb = get_snapshot(conn, bid)
    if not sa or not sb:
        return jsonify({'error': 'snapshot not found'}), 404
    # Align by suspect id
    map_a = {s['id']: s for s in sa['payload']}
    map_b = {s['id']: s for s in sb['payload']}
    all_ids = sorted(set(map_a.keys()) | set(map_b.keys()))
    diffs = []
    for sid in all_ids:
        ea = map_a.get(sid) or {}
        eb = map_b.get(sid) or {}
        diffs.append({
            'id': sid,
            'name': ea.get('name') or eb.get('name'),
            'a_composite': ea.get('composite_score'),
            'b_composite': eb.get('composite_score'),
            'delta': (None if (ea.get('composite_score') is None or eb.get('composite_score') is None) else (eb.get('composite_score') - ea.get('composite_score')))
        })
    return jsonify({'a': sa['id'], 'b': sb['id'], 'diffs': diffs, 'label_a': sa.get('label'), 'label_b': sb.get('label'), 'created_a': sa.get('created_at'), 'created_b': sb.get('created_at')})


@app.get('/api/suspects/<sid>/waterfall')
def api_score_waterfall(sid: str):
    case_id = request.args.get('case_id') or 'default'
    with get_conn() as conn:
        s = db_get_suspect(conn, sid)
        if not s:
            return jsonify({'error': 'not found'}), 404
    # The components already persisted on suspect row
    # Provide an ordered breakdown for front-end waterfall
    ml = s.get('score') or 0.0
    ev = s.get('evidence_score') or 0.0
    offense_boost = s.get('offense_boost') or 0.0
    base = s.get('composite_base') if s.get('composite_base') is not None else (0.7*ml + 0.3*ev)  # fallback
    comp = s.get('composite_score') or (base + offense_boost)
    steps = [
        {'label': 'ML Score', 'value': ml, 'type': 'base'},
        {'label': 'Evidence Score', 'value': ev, 'type': 'increment'},
        {'label': 'Weighted Base', 'value': base, 'type': 'result'},
        {'label': 'Offense Boost', 'value': offense_boost, 'type': 'increment'},
        {'label': 'Composite', 'value': comp, 'type': 'total'}
    ]
    return jsonify({'suspect_id': sid, 'case_id': case_id, 'steps': steps})


@app.post('/api/clues/duplicates/merge')
@auth_required
def api_merge_duplicates():
    data = request.get_json(force=True) or {}
    canonical_id = data.get('canonical_id')
    duplicate_ids = data.get('duplicate_ids') or []
    if not canonical_id or not isinstance(duplicate_ids, list) or not duplicate_ids:
        return jsonify({'error': 'canonical_id and duplicate_ids required'}), 400
    with get_conn() as conn:
        cur = conn.cursor()
        # ensure canonical exists
        cur.execute("SELECT id FROM clues WHERE id=?", (canonical_id,))
        if not cur.fetchone():
            return jsonify({'error': 'canonical clue not found'}), 404
        # re-point duplicates: strategy = delete duplicates (optionally could keep and mark)
        deleted = 0
        for did in duplicate_ids:
            try:
                cur.execute("DELETE FROM clues WHERE id=? AND duplicate_of_id=?", (did, canonical_id))
                if cur.rowcount:
                    deleted += 1
            except Exception:
                pass
        conn.commit()
    return jsonify({'ok': True, 'deleted_duplicates': deleted, 'canonical_id': canonical_id})


@app.post('/api/clues/duplicates/delete')
@auth_required
def api_delete_duplicates():
    data = request.get_json(force=True) or {}
    ids = data.get('ids') or []
    if not isinstance(ids, list) or not ids:
        return jsonify({'error': 'ids list required'}), 400
    with get_conn() as conn:
        cur = conn.cursor()
        deleted = 0
        for cid in ids:
            try:
                cur.execute("DELETE FROM clues WHERE id=?", (cid,))
                if cur.rowcount:
                    deleted += 1
            except Exception:
                pass
        conn.commit()
    return jsonify({'ok': True, 'deleted': deleted})


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
    try:
        publish_event('evidence_added', {'suspect_id': sid, 'case_id': data.get('case_id','default')})
    except Exception:
        pass
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


@app.post('/api/analysis/pdf')
def api_case_analysis_pdf():
    """Generate a PDF report for a case analysis and return it as an attachment.

    Body JSON: { case_id?: str, style?: 'brief'|'detailed' }
    """
    data = request.get_json(silent=True) or {}
    case_id = data.get('case_id') or request.args.get('case_id') or 'default'
    style = data.get('style', 'brief')
    with get_conn() as conn:
        if case_id and not get_case(conn, case_id):
            return jsonify({'error': 'case not found'}), 404
        suspects = db_list_suspects(conn, case_id)
        clues_rows = db_list_clues(conn, case_id=case_id)
        clues = [r['text'] for r in clues_rows]
    report = generate_case_analysis(case_id, clues, suspects, style=style)
    # Prepare printable deductions: split report text into lines for PDF
    deductions = (report.get('report') or '').splitlines()
    out_dir = BASE_DIR / 'outputs'
    out_dir.mkdir(exist_ok=True)
    import time
    fname = f"case_{case_id}_analysis_{int(time.time())}.pdf"
    out_path = out_dir / fname
    try:
        save_pdf_report(clues, deductions, str(out_path))
    except Exception as e:
        return jsonify({'error': f'pdf generation failed: {e}'}), 500
    try:
        return send_file(str(out_path), as_attachment=True, download_name=fname, mimetype='application/pdf')
    except Exception as e:
        return jsonify({'error': f'failed to send pdf: {e}'}), 500


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
    try:
        publish_event('feedback_added', {'suspect_id': suspect_id, 'decision': decision, 'case_id': case_id})
    except Exception:
        pass
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
    # Compose context from chunks and clues (top few for token budget)
    context_lines = []
    for ch in ranked_chunks[:5]:
        context_lines.append(f"CHUNK: {ch['text']}")
    for c in clue_hits[:5]:
        context_lines.append(f"CLUE: {c['text']}")
    context_text = "\n".join(context_lines)
    # Try LLM-backed answer
    try:
        ans = answer_with_context(q, context_text)
        backend = ans.get('backend','heuristic')
        answer = ans.get('answer','')
    except Exception:
        backend = 'heuristic'
        # Heuristic: overlap-based extraction as fallback
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
        'backend': backend,
        'chunks': ranked_chunks,
        'clues': clue_hits,
        'case_id': case_id
    })


@app.get('/api/qa/advanced')
def api_qa_advanced():
    """Advanced hybrid RAG retrieval endpoint.

    Query params:
      q: question (required)
      case_id: scope (default 'default')
      k: desired diversified context size (default 10)
    Returns: diversified context candidates and echo of query.
    """
    q = request.args.get('q')
    if not q:
        return jsonify({'error': 'q required'}), 400
    case_id = request.args.get('case_id') or 'default'
    try:
        k = int(request.args.get('k','10'))
    except ValueError:
        k = 10
    out = advanced_retrieve(q, case_id=case_id, mmr_k=k)
    return jsonify(out)


@app.post('/api/chat')
def api_chat():
    """Streaming chat-like endpoint.

    Body JSON: { question: str, case_id?: str }
    Returns: text/plain stream of the answer, chunked.
    """
    data = request.get_json(silent=True) or {}
    q = data.get('question') or data.get('q')
    case_id = data.get('case_id') or 'default'
    if not q:
        return jsonify({'error': 'question required'}), 400
    # Build compact context (reuse QA logic)
    try:
        k = 5
        with get_conn() as conn:
            chunks = list_chunks(conn, case_id, limit=5000)
        ranked_chunks = []
        if chunks and any(c.get('embedding') for c in chunks):
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                _model_q = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                q_vec = _model_q.encode([q], normalize_embeddings=True)[0]  # type: ignore
                import numpy as np
                def cos(a,b):
                    a=np.array(a); b=np.array(b)
                    return float((a@b)/( (a.dot(a)**0.5)*(b.dot(b)**0.5)+1e-9 ))
                scored = []
                for c in chunks:
                    emb = c.get('embedding')
                    if not emb: continue
                    try:
                        score = cos(q_vec, emb)
                    except Exception:
                        score = 0.0
                    scored.append((score, c))
                scored.sort(key=lambda x: x[0], reverse=True)
                ranked_chunks = [ {'text': sc[1]['text']} for sc in scored[:k] ]
            except Exception:
                ranked_chunks = []
        clue_hits = semantic_search(q, k=k, case_id=case_id)
        context_lines = []
        for ch in ranked_chunks[:5]:
            context_lines.append(f"CHUNK: {ch['text']}")
        for c in clue_hits[:5]:
            context_lines.append(f"CLUE: {c['text']}")
        context_text = "\n".join(context_lines)
    except Exception:
        context_text = ''

    def _gen():
        try:
            for chunk in stream_answer(q, context_text):
                yield chunk
        except Exception:
            yield 'Unable to generate answer right now.\n'

    return Response(stream_with_context(_gen()), mimetype='text/plain; charset=utf-8')


@app.get('/api/events/stream')
def api_events_stream():
    """Server-Sent Events stream for realtime updates (jobs, clues, feedback)."""
    return Response(stream_with_context(sse_stream_generator()), mimetype='text/event-stream')


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


@app.get('/api/graph/analytics')
def api_graph_analytics():
    case_id = request.args.get('case_id') or 'default'
    graph = analyze_graph(case_id)
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
    # Embedding / semantic index metrics (best-effort; absent if not initialized yet)
    try:
        emb_metrics = semantic_embedding_metrics()
    except Exception:
        emb_metrics = None
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
        'embeddings': emb_metrics,
        'data_freshness': ls,
        'case_id': case_id or 'ALL'
    })


@app.get('/prometheus/metrics')
def prometheus_metrics():  # lightweight export without global registry instrumentation
    """Expose selected metrics in Prometheus text format.

    For full instrumentation integrate prometheus_client and counters; this is a minimal snapshot.
    """
    lines = []
    def emit(name: str, val: Any, help_text: str):
        if val is None:
            return
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        try:
            lines.append(f"{name} {float(val)}")
        except Exception:
            pass
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as c FROM suspects")
            emit('ai_detective_suspects_total', cur.fetchone()['c'], 'Total suspects')
            cur.execute("SELECT COUNT(*) as c FROM clues")
            emit('ai_detective_clues_total', cur.fetchone()['c'], 'Total clues')
            cur.execute("SELECT COUNT(*) as c FROM documents")
            emit('ai_detective_documents_total', cur.fetchone()['c'], 'Total documents')
        from src.semantic_search import get_embedding_metrics as _gem  # type: ignore
        em = _gem()
        emit('ai_detective_embedding_cases', em.get('total_cases_indexed'), 'Cases with built embedding index')
        emit('ai_detective_embedding_clues', em.get('total_clues_indexed'), 'Clues indexed in embeddings')
        emit('ai_detective_last_refresh_ms', em.get('last_refresh_duration_ms'), 'Last embedding refresh duration milliseconds')
    except Exception:
        pass
    body = '\n'.join(lines) + '\n'
    return Response(body, mimetype='text/plain; charset=utf-8')


if __name__ == "__main__":
    # Helpful startup diagnostics: list registered routes once.
    print("\n[AI Detective] Registered routes:")
    try:
        for r in app.url_map.iter_rules():
            if str(r).startswith('/static'):
                continue
            methods = ','.join(sorted(m for m in r.methods if m not in {'HEAD','OPTIONS'}))
            print(f"  {r}  ->  {methods}")
    except Exception:
        pass
    # Add a lightweight health endpoint dynamically if not already present
    if 'health' not in {str(r).strip('/') for r in app.url_map.iter_rules()}:
        @app.get('/health')
        def health():  # type: ignore
            return jsonify({'ok': True, 'service': 'ai-detective', 'routes': len(list(app.url_map.iter_rules()))})

    # JSON 404 handler to aid frontend debugging (returns path + hint)
    @app.errorhandler(404)
    def _not_found(e):  # type: ignore
        from flask import request as _req
        return jsonify({'error': 'not found', 'path': _req.path, 'hint': 'If expected /api/* routes are missing, ensure you are running src/api.py from project root.'}), 404

    import os
    port = int(os.environ.get('AI_DETECTIVE_PORT', '5000'))
    print(f"\n[AI Detective] Starting server on 0.0.0.0:{port} (set AI_DETECTIVE_PORT env var to change)\n")
    # Optional background scheduler (simple thread) for periodic embeddings refresh
    if os.environ.get('ENABLE_SCHEDULER','0') in {'1','true','yes'}:
        import threading
        from src.jobs_backend import start_job, task_embeddings_refresh  # type: ignore
        interval = int(os.environ.get('SCHEDULER_REFRESH_INTERVAL_SEC','3600'))
        def _loop():
            while True:
                try:
                    start_job('embeddings_refresh', task_embeddings_refresh)
                except Exception:
                    pass
                time.sleep(interval)
        threading.Thread(target=_loop, name='embeddings-scheduler', daemon=True).start()
    app.run(host="0.0.0.0", port=port, debug=True)
