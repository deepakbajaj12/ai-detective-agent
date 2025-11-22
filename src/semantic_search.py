"""Semantic (or lexical) search over clues.

Design:
 - Tries to use SentenceTransformer (MiniLM) if available for dense embeddings.
 - Falls back to TF-IDF vector space cosine similarity if transformers not installed.
 - Caches per-case index (vectors + clue metadata) in memory; can be invalidated.
 - Intended for small/medium datasets; for larger scale swap with FAISS / Chroma.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import math
import threading

try:  # support both package and script execution
    from src.db import get_conn, list_clues, upsert_clue_embedding, list_case_embeddings  # type: ignore
except Exception:
    from db import get_conn, list_clues, upsert_clue_embedding, list_case_embeddings  # type: ignore

_lock = threading.Lock()
_indexes: Dict[str, Dict[str, Any]] = {}
_embedding_mode: Optional[str] = None  # 'dense' | 'tfidf'
# Embedding metrics (lightweight, in-memory)
_metrics: Dict[str, Any] = {
    'total_cases_indexed': 0,
    'total_clues_indexed': 0,
    'last_refresh_started': None,
    'last_refresh_duration_ms': None,
    'backend': None,
}

try:  # Attempt dense embedding backend
    from sentence_transformers import SentenceTransformer  # type: ignore
    _model: Optional[SentenceTransformer] = None
    _embedding_mode = 'dense'
except Exception:  # Fallback to TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _model = None
    _embedding_mode = 'tfidf'


def _ensure_model():
    global _model
    if _embedding_mode == 'dense' and _model is None:  # lazy load
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def _l2_normalize(vecs):
    import numpy as np
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def build_index(case_id: str = 'default', force: bool = False) -> None:
    """Build or rebuild the in-memory index for a case.

    Persistence strategy:
      - If dense backend and stored embeddings exist & count matches clues, load them instead of recomputing.
      - After computing fresh dense embeddings, upsert each embedding row to the clue_embeddings table.
      - TF-IDF backend is transient (not persisted) due to vocabulary dependence; recomputed each build.
    """
    with _lock:
        if not force and case_id in _indexes:
            return
        with get_conn() as conn:
            clues = list_clues(conn, case_id=case_id)
            stored = list_case_embeddings(conn, case_id) if _embedding_mode == 'dense' else []
        texts = [c['text'] for c in clues]
        if not texts:
            _indexes[case_id] = {"clues": [], "vectors": None, "backend": _embedding_mode}
            return
        if _embedding_mode == 'dense':
            _ensure_model()
            # Attempt reuse
            if stored and len(stored) == len(clues):
                # Ensure ordering by clue_id aligns with clues list ordering (both ascending id assumption)
                # Map by clue_id for safety
                emb_map = {r['clue_id']: r.get('embedding') for r in stored}
                vectors = []
                for c in clues:
                    v = emb_map.get(c['id'])
                    if v is None:
                        vectors = []
                        break
                    vectors.append(v)
                if vectors:
                    import numpy as np
                    vectors = np.array(vectors)
                    _indexes[case_id] = {"clues": clues, "vectors": vectors, "backend": _embedding_mode}
                    _metrics['total_cases_indexed'] = len(_indexes)
                    _metrics['total_clues_indexed'] = sum(len(v.get('clues') or []) for v in _indexes.values())
                    _metrics['backend'] = _embedding_mode
                    return
            # Fresh compute
            vectors = _model.encode(texts, normalize_embeddings=True)  # type: ignore
            # Persist each embedding row (list form)
            try:
                with get_conn() as conn:
                    for emb, clue in zip(vectors.tolist(), clues):  # type: ignore
                        try:
                            upsert_clue_embedding(conn, clue['id'], case_id, _embedding_mode or 'dense', emb)
                        except Exception:
                            pass
            except Exception:
                pass
            _indexes[case_id] = {"clues": clues, "vectors": vectors, "backend": _embedding_mode}
        else:  # TF-IDF fallback (transient)
            vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
            mat = vectorizer.fit_transform(texts)
            _indexes[case_id] = {
                "clues": clues,
                "vectors": mat,
                "backend": _embedding_mode,
                "vectorizer": vectorizer
            }
        _metrics['total_cases_indexed'] = len(_indexes)
        _metrics['total_clues_indexed'] = sum(len(v.get('clues') or []) for v in _indexes.values())
        _metrics['backend'] = _embedding_mode


def search(query: str, k: int = 5, case_id: str = 'default') -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []
    if case_id not in _indexes:
        build_index(case_id)
    idx = _indexes.get(case_id)
    if not idx or not idx.get('clues'):
        return []
    backend = idx['backend']
    if backend == 'dense':
        _ensure_model()
        q_vec = _model.encode([query], normalize_embeddings=True)  # type: ignore
        import numpy as np
        doc_vecs = idx['vectors']  # (N, D)
        sims = (doc_vecs @ q_vec.T).ravel()
    else:  # TF-IDF
        vectorizer = idx['vectorizer']
        mat = idx['vectors']
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(mat, q_vec).ravel()
    # rank
    import numpy as np
    order = np.argsort(sims)[::-1][:k]
    results: List[Dict[str, Any]] = []
    for i in order:
        score = float(sims[i])
        if math.isfinite(score):
            c = idx['clues'][int(i)]
            results.append({
                "clue_id": c['id'],
                "text": c['text'],
                "suspect_id": c.get('suspect_id'),
                "score": round(score, 6)
            })
    return results


def refresh(case_id: str = 'default') -> None:
    import time
    start = time.time()
    if _metrics['last_refresh_started'] is None:
        _metrics['last_refresh_started'] = start
    build_index(case_id, force=True)
    dur = (time.time() - start) * 1000.0
    _metrics['last_refresh_duration_ms'] = dur


def backend_mode() -> str:
    return _embedding_mode or 'unknown'


def refresh_all(case_ids: List[str]) -> Dict[str, Any]:  # utility for job task
    import time
    started = time.time()
    for cid in case_ids:
        refresh(cid)
    total_ms = (time.time() - started) * 1000.0
    return {
        'cases': len(case_ids),
        'duration_ms': total_ms,
        'backend': backend_mode(),
        'total_clues_indexed': _metrics.get('total_clues_indexed')
    }


def get_embedding_metrics() -> Dict[str, Any]:
    return {
        'backend': _metrics.get('backend'),
        'total_cases_indexed': _metrics.get('total_cases_indexed'),
        'total_clues_indexed': _metrics.get('total_clues_indexed'),
        'last_refresh_started': _metrics.get('last_refresh_started'),
        'last_refresh_duration_ms': _metrics.get('last_refresh_duration_ms'),
    }
