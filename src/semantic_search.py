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
    from src.db import get_conn, list_clues  # type: ignore
except Exception:
    from db import get_conn, list_clues  # type: ignore

_lock = threading.Lock()
_indexes: Dict[str, Dict[str, Any]] = {}
_embedding_mode: Optional[str] = None  # 'dense' | 'tfidf'

try:  # Attempt dense embedding backend
    from sentence_transformers import SentenceTransformer
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
    """Build or rebuild the in-memory index for a case."""
    with _lock:
        if not force and case_id in _indexes:
            return
        with get_conn() as conn:
            clues = list_clues(conn, case_id=case_id)
        texts = [c['text'] for c in clues]
        if not texts:
            _indexes[case_id] = {"clues": [], "vectors": None, "backend": _embedding_mode}
            return
        if _embedding_mode == 'dense':
            _ensure_model()
            vectors = _model.encode(texts, normalize_embeddings=True)  # type: ignore
        else:  # TF-IDF fallback
            vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
            mat = vectorizer.fit_transform(texts)
            # Store components to compute query vector
            _indexes[case_id] = {
                "clues": clues,
                "vectors": mat,
                "backend": _embedding_mode,
                "vectorizer": vectorizer
            }
            return
        _indexes[case_id] = {"clues": clues, "vectors": vectors, "backend": _embedding_mode}


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
    build_index(case_id, force=True)


def backend_mode() -> str:
    return _embedding_mode or 'unknown'
