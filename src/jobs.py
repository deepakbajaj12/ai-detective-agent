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
        except Exception as e:  # pragma: no cover (best-effort)
            with _lock:
                _jobs[job_id]['status'] = 'failed'
                _jobs[job_id]['error'] = str(e)
                _jobs[job_id]['updated_at'] = _now_ts()

    t = threading.Thread(target=runner, name=f"job-{job_type}-{job_id[:6]}", daemon=True)
    t.start()
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

