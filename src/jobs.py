from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def _now_ts() -> float:
    return time.time()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return _jobs.get(job_id)


def start_job(job_type: str, target: Callable[[Callable[[int, Optional[str]], None]], Any], *args, **kwargs) -> str:
    """
    Start a background job. The target must accept the first argument as a
    progress callback: set_progress(percent:int, message:str|None).
    Returns a job_id.
    """
    job_id = uuid.uuid4().hex
    job = {
        'id': job_id,
        'type': job_type,
        'status': 'queued',  # queued|running|completed|failed
        'progress': 0,
        'message': None,
        'result': None,
        'error': None,
        'created_at': _now_ts(),
        'updated_at': _now_ts(),
    }
    with _lock:
        _jobs[job_id] = job

    def set_progress(p: int, msg: Optional[str] = None):
        with _lock:
            j = _jobs.get(job_id)
            if not j:
                return
            j['progress'] = max(0, min(100, int(p)))
            if msg is not None:
                j['message'] = msg
            j['updated_at'] = _now_ts()
        # Publish realtime progress event
        try:
            from .events_bus import publish_event  # type: ignore
            publish_event('job_progress', {
                'job_id': job_id,
                'type': job_type,
                'progress': int(p),
                'message': msg
            })
        except Exception:
            pass

    def runner():
        with _lock:
            _jobs[job_id]['status'] = 'running'
            _jobs[job_id]['updated_at'] = _now_ts()
        try:
            result = target(set_progress, *args, **kwargs)
            set_progress(100)
            with _lock:
                _jobs[job_id]['status'] = 'completed'
                _jobs[job_id]['result'] = result
                _jobs[job_id]['updated_at'] = _now_ts()
            try:
                from .events_bus import publish_event  # type: ignore
                publish_event('job_completed', {
                    'job_id': job_id,
                    'type': job_type,
                    'result': result
                })
            except Exception:
                pass
        except Exception as e:  # pragma: no cover (best-effort)
            with _lock:
                _jobs[job_id]['status'] = 'failed'
                _jobs[job_id]['error'] = str(e)
                _jobs[job_id]['updated_at'] = _now_ts()
            try:
                from .events_bus import publish_event  # type: ignore
                publish_event('job_failed', {
                    'job_id': job_id,
                    'type': job_type,
                    'error': str(e)
                })
            except Exception:
                pass

    t = threading.Thread(target=runner, name=f"job-{job_type}-{job_id[:6]}", daemon=True)
    t.start()
    # Publish job queued event
    try:
        from .events_bus import publish_event  # type: ignore
        publish_event('job_queued', {
            'job_id': job_id,
            'type': job_type
        })
    except Exception:
        pass
    return job_id


# ---- Built-in job tasks ----

def task_transformer_train(set_progress: Callable[[int, Optional[str]], None], training_json: str | Path) -> Dict[str, Any]:
    from .ml_transformer import ensure_transformer_model
    p = Path(training_json)
    set_progress(5, 'validating input')
    if not p.exists():
        raise FileNotFoundError(f"training data not found: {p}")
    set_progress(15, 'starting training')
    ensure_transformer_model(str(p))
    set_progress(95, 'finalizing')
    return {'ok': True, 'path': str(p)}


def task_index_refresh(set_progress: Callable[[int, Optional[str]], None], case_id: str = 'default') -> Dict[str, Any]:
    try:
        from .semantic_search import refresh
    except Exception:  # when executed as script
        from semantic_search import refresh  # type: ignore
    set_progress(10, f'refreshing index for case {case_id}')
    refresh(case_id)
    set_progress(100, 'done')
    return {'ok': True, 'case_id': case_id}


def task_embeddings_refresh(set_progress: Callable[[int, Optional[str]], None]) -> Dict[str, Any]:
    """Refresh embeddings for all cases (rebuild indexes sequentially)."""
    try:
        from .db import get_conn, list_cases
        from .semantic_search import refresh_all
    except Exception:
        from db import get_conn, list_cases  # type: ignore
        from semantic_search import refresh_all  # type: ignore
    set_progress(5, 'enumerating cases')
    with get_conn() as conn:
        cases = list_cases(conn)
    case_ids = [c['id'] for c in cases] or ['default']
    total = len(case_ids)
    if total == 0:
        set_progress(100, 'no cases found')
        return {'ok': True, 'cases': 0}
    # Iterate and update progress
    for i, cid in enumerate(case_ids, start=1):
        pct = int(5 + (90 * (i/total)))
        set_progress(pct, f'refresh {cid}')
        try:
            # reuse single-case refresh for more granular duration tracking
            from .semantic_search import refresh  # type: ignore
            refresh(cid)
        except Exception:
            continue
    from .semantic_search import get_embedding_metrics  # type: ignore
    metrics = get_embedding_metrics()
    set_progress(100, 'embeddings refreshed')
    return {'ok': True, 'cases': total, 'metrics': metrics}

