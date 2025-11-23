"""Advanced Retrieval-Augmented Generation helpers.

Hybrid retrieval strategy:
 1. Dense similarity over document chunks embeddings (if available).
 2. Semantic clue search via existing semantic_search backend.
 3. Lexical TF-IDF/BM25-lite scoring (approx) on clues for query terms.
 4. Merge candidates and apply Maximal Marginal Relevance (MMR) to promote diversity.

Outputs a ranked list of context snippets with provenance metadata suitable for downstream answer generation.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import math

try:
    from src.db import get_conn, list_chunks, list_clues
    from src.semantic_search import search as semantic_search
except Exception:
    from db import get_conn, list_chunks, list_clues  # type: ignore
    from semantic_search import search as semantic_search  # type: ignore


def _lexical_score(query: str, text: str) -> float:
    import re
    q_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", query)]
    if not q_tokens:
        return 0.0
    t_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)]
    if not t_tokens:
        return 0.0
    freq = {}
    for t in t_tokens:
        freq[t] = freq.get(t, 0) + 1
    score = 0.0
    for qt in q_tokens:
        if qt in freq:
            score += 1.0 + math.log(1 + freq[qt])
    return score / len(q_tokens)


def _mmr(query_vec, candidates: List[Tuple[str, float, Dict[str, Any]]], lambda_param: float = 0.7, top_k: int = 10) -> List[Dict[str, Any]]:
    """Apply MMR to candidate list.
    candidates: list of (id, relevance_score, metadata + 'vector')
    metadata must include 'vector' for dense backend or will fallback to relevance only.
    """
    import numpy as np
    selected: List[Dict[str, Any]] = []
    candidate_pool = candidates.copy()
    while candidate_pool and len(selected) < top_k:
        best_idx = None
        best_score = -1e9
        for i, (cid, rel, meta) in enumerate(candidate_pool):
            vec = meta.get('vector')
            if selected and vec is not None:
                max_sim = max(_cos(vec, s.get('vector')) for s in selected if s.get('vector') is not None) if any(s.get('vector') is not None for s in selected) else 0.0
            else:
                max_sim = 0.0
            mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        cid, rel, meta = candidate_pool.pop(best_idx)
        out = meta.copy()
        out['mmr_score'] = best_score
        out['relevance'] = rel
        selected.append(out)
    # Drop raw vectors from final output to reduce payload size
    for s in selected:
        if 'vector' in s:
            s.pop('vector')
    return selected


def _cos(a, b) -> float:
    if a is None or b is None:
        return 0.0
    import numpy as np
    a = np.array(a); b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float((a @ b) / denom)


def advanced_retrieve(query: str, case_id: str = 'default', top_chunks: int = 8, top_clues: int = 8, mmr_k: int = 10) -> Dict[str, Any]:
    """Perform hybrid retrieval for a query returning diversified context snippets."""
    if not query.strip():
        return {'query': query, 'results': []}
    # 1. Chunk embeddings cosine ranking (reuse existing embedding logic)
    with get_conn() as conn:
        chunks = list_chunks(conn, case_id, limit=5000)
        clues_rows = list_clues(conn, case_id=case_id)
    chunk_candidates: List[Tuple[str, float, Dict[str, Any]]] = []
    import numpy as np
    q_vec = None
    if chunks and any(c.get('embedding') for c in chunks):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _q_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            q_vec = _q_model.encode([query], normalize_embeddings=True)[0]  # type: ignore
            for ch in chunks:
                emb = ch.get('embedding')
                if not emb:
                    continue
                score = _cos(q_vec, emb)
                chunk_candidates.append((f"chunk:{ch['id']}", score, {
                    'id': ch['id'],
                    'type': 'chunk',
                    'text': ch['text'],
                    'vector': emb,
                    'raw_score': score
                }))
        except Exception:
            pass
    chunk_candidates.sort(key=lambda x: x[1], reverse=True)
    chunk_candidates = chunk_candidates[:top_chunks]
    # 2. Semantic clue search
    clue_sem = semantic_search(query, k=top_clues, case_id=case_id)
    clue_sem_candidates: List[Tuple[str, float, Dict[str, Any]]] = []
    for c in clue_sem:
        clue_sem_candidates.append((f"clue_sem:{c['clue_id']}", c['score'], {
            'id': c['clue_id'],
            'type': 'clue_sem',
            'text': c['text'],
            'vector': None,
            'raw_score': c['score']
        }))
    # 3. Lexical scoring on clue corpus
    lexical_candidates: List[Tuple[str, float, Dict[str, Any]]] = []
    for r in clues_rows[:2000]:  # cap for efficiency
        ls = _lexical_score(query, r['text'])
        if ls <= 0:
            continue
        lexical_candidates.append((f"clue_lex:{r['id']}", ls, {
            'id': r['id'],
            'type': 'clue_lex',
            'text': r['text'],
            'vector': None,
            'raw_score': ls
        }))
    lexical_candidates.sort(key=lambda x: x[1], reverse=True)
    lexical_candidates = lexical_candidates[:top_clues]
    # Merge all
    combined = chunk_candidates + clue_sem_candidates + lexical_candidates
    # Normalize relevance scores to 0..1 for fairness
    if combined:
        scores = [c[1] for c in combined]
        mx = max(scores) or 1.0
        mn = min(scores)
        normed = []
        for cid, sc, meta in combined:
            if mx == mn:
                rel = 0.5
            else:
                rel = (sc - mn) / (mx - mn)
            normed.append((cid, rel, meta))
        combined = normed
    diversified = _mmr(q_vec, combined, lambda_param=0.7, top_k=mmr_k)
    return {
        'query': query,
        'case_id': case_id,
        'candidates': diversified,
        'raw_counts': {
            'chunks_considered': len(chunks),
            'chunk_hits': len(chunk_candidates),
            'semantic_clue_hits': len(clue_sem_candidates),
            'lexical_clue_hits': len(lexical_candidates)
        }
    }
