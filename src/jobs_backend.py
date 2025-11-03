from __future__ import annotations

import os
from typing import Any, Dict, Optional, List, Callable

# In-memory fallback
from . import jobs as mem_jobs

# Optional RQ backend
try:
    from . import jobs_rq as rq_jobs
except Exception:
    rq_jobs = None  # type: ignore


def _rq_available() -> bool:
    if rq_jobs is None:
        return False
    # Allow env var to force disable RQ
    if os.environ.get('USE_RQ', '').lower() in {'0', 'false', 'no'}:
        return False
    try:
        conn = rq_jobs._redis_connection()  # type: ignore
        return bool(conn)
    except Exception:
        return False


def backend_mode() -> str:
    return 'rq' if _rq_available() else 'memory'


def start_job(job_type: str, target: Callable[..., Any] | None, *args, **kwargs) -> str:
    """Unified start_job.

    For RQ, we ignore `target` and map `job_type` to the appropriate enqueued function.
    For memory, we defer to mem_jobs.start_job and pass through the `target` and args.
    """
    if _rq_available():
        if job_type == 'index_refresh':
            jid = rq_jobs.start_index_refresh(*args, **kwargs)  # type: ignore
        elif job_type == 'transformer_train':
            jid = rq_jobs.start_transformer_train(*args, **kwargs)  # type: ignore
        else:
            raise ValueError(f"Unknown job_type for RQ: {job_type}")
        if not jid:
            raise RuntimeError('Failed to enqueue RQ job')
        return jid
    # fallback
    return mem_jobs.start_job(job_type, target, *args, **kwargs)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    if _rq_available():
        j = rq_jobs.get_job(job_id)  # type: ignore
        return j
    return mem_jobs.get_job(job_id)


def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    if _rq_available():
        return rq_jobs.list_jobs(limit)  # type: ignore
    # synthesize history from in-memory registry
    from time import time
    rows = []
    for j in mem_jobs._jobs.values():  # type: ignore
        rows.append({
            'id': j['id'],
            'status': j['status'],
            'progress': j.get('progress', 0),
            'message': j.get('message'),
            'created_at': j.get('created_at') or time(),
            'ended_at': j.get('updated_at'),
            'func': j.get('type'),
        })
    rows.sort(key=lambda x: x.get('created_at') or 0, reverse=True)
    return rows[:limit]


def cancel_job(job_id: str) -> bool:
    if _rq_available():
        return rq_jobs.cancel_job(job_id)  # type: ignore
    # not supported for in-memory
    return False


# Re-export task functions for convenience
from .jobs import task_transformer_train, task_index_refresh  # noqa: E402,F401
