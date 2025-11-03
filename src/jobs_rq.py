from __future__ import annotations

import os
from typing import Any, Dict, Optional, List

try:
    from redis import Redis
    from rq import Queue
    from rq.job import Job
    from rq.registry import StartedJobRegistry, FinishedJobRegistry, FailedJobRegistry, ScheduledJobRegistry, DeferredJobRegistry
    from rq import get_current_job
except Exception:  # pragma: no cover
    # Allow import even if rq/redis not installed; callers must guard usage.
    Redis = None  # type: ignore
    Queue = None  # type: ignore
    Job = None  # type: ignore
    StartedJobRegistry = FinishedJobRegistry = FailedJobRegistry = ScheduledJobRegistry = DeferredJobRegistry = None  # type: ignore
    def get_current_job():  # type: ignore
        return None


def _redis_connection() -> Optional[Redis]:
    url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    if Redis is None:
        return None
    try:
        conn = Redis.from_url(url)
        # ping to verify connectivity
        conn.ping()
        return conn
    except Exception:
        return None


def _queue(conn: Redis) -> Queue:
    return Queue('default', connection=conn)


# ---- RQ task wrappers that update progress on job.meta ----

def _make_set_progress():
    def set_progress(pct: int, msg: Optional[str] = None):
        try:
            job = get_current_job()  # type: ignore
            if job is None:
                return
            meta = job.meta or {}
            meta['progress'] = int(max(0, min(100, pct)))
            if msg is not None:
                meta['message'] = msg
            job.meta = meta
            job.save_meta()
        except Exception:
            pass
    return set_progress


def task_index_refresh_rq(case_id: str = 'default') -> Dict[str, Any]:
    from .jobs import task_index_refresh  # reuse core task logic
    set_prog = _make_set_progress()
    set_prog(5, f'start index refresh for {case_id}')
    result = task_index_refresh(set_prog, case_id)
    set_prog(100, 'done')
    return result


def task_transformer_train_rq(training_json: str) -> Dict[str, Any]:
    from .jobs import task_transformer_train
    set_prog = _make_set_progress()
    set_prog(5, 'validate training data')
    result = task_transformer_train(set_prog, training_json)
    set_prog(100, 'done')
    return result


# ---- Public API used by jobs_backend when RQ is enabled ----

def start_index_refresh(case_id: str = 'default') -> Optional[str]:
    conn = _redis_connection()
    if not conn:
        return None
    q = _queue(conn)
    job = q.enqueue(task_index_refresh_rq, case_id, job_timeout=3600)
    return job.get_id()


def start_transformer_train(training_json: str) -> Optional[str]:
    conn = _redis_connection()
    if not conn:
        return None
    q = _queue(conn)
    job = q.enqueue(task_transformer_train_rq, training_json, job_timeout=24*3600)
    return job.get_id()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = _redis_connection()
    if not conn:
        return None
    try:
        job = Job.fetch(job_id, connection=conn)
    except Exception:
        return None
    meta = job.meta or {}
    out = {
        'id': job.get_id(),
        'status': job.get_status(refresh=False),
        'progress': int(meta.get('progress', 0) or 0),
        'message': meta.get('message'),
        'result': job.result if job.is_finished else None,
        'error': (job.exc_info or None),
        'created_at': job.created_at.timestamp() if job.created_at else None,
        'enqueued_at': job.enqueued_at.timestamp() if job.enqueued_at else None,
        'ended_at': job.ended_at.timestamp() if job.ended_at else None,
        'origin': job.origin,
    }
    return out


def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = _redis_connection()
    if not conn:
        return []
    q = _queue(conn)
    registries = [
        StartedJobRegistry('default', connection=conn),
        FinishedJobRegistry('default', connection=conn),
        FailedJobRegistry('default', connection=conn),
        ScheduledJobRegistry('default', connection=conn),
        DeferredJobRegistry('default', connection=conn),
    ]
    ids = []
    for reg in registries:
        try:
            ids.extend(reg.get_job_ids()[:limit])
        except Exception:
            pass
    # de-duplicate while preserving order
    seen = set()
    ordered = []
    for jid in ids:
        if jid in seen:
            continue
        seen.add(jid)
        ordered.append(jid)
    items: List[Dict[str, Any]] = []
    for jid in ordered[:limit]:
        try:
            j = Job.fetch(jid, connection=conn)
            meta = j.meta or {}
            items.append({
                'id': j.get_id(),
                'status': j.get_status(refresh=False),
                'progress': int(meta.get('progress', 0) or 0),
                'message': meta.get('message'),
                'created_at': j.created_at.timestamp() if j.created_at else None,
                'ended_at': j.ended_at.timestamp() if j.ended_at else None,
                'func': j.func_name,
            })
        except Exception:
            continue
    # naive sort: newest first by created_at
    items.sort(key=lambda x: x.get('created_at') or 0, reverse=True)
    return items


def cancel_job(job_id: str) -> bool:
    conn = _redis_connection()
    if not conn:
        return False
    try:
        j = Job.fetch(job_id, connection=conn)
        j.cancel()
        return True
    except Exception:
        try:
            q = _queue(conn)
            q.cancel_job(job_id)
            return True
        except Exception:
            return False
