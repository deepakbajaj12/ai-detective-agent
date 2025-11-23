"""In-memory simple event bus for realtime server-sent events (SSE).

Design:
 - Maintains a registry of subscriber queues (one per active connection).
 - `publish_event` broadcasts an event dict to all subscriber queues.
 - SSE endpoint consumes events from a dedicated queue until client disconnects.
 - Heartbeats emitted if idle to keep connection alive.
Limitations: In-memory only; events lost on restart; no persistence or replay beyond connection lifetime.
"""
from __future__ import annotations
import threading
import time
from typing import Dict, Any, List
from queue import Queue, Empty

_sub_lock = threading.Lock()
_subscribers: List[Queue] = []

def publish_event(event_type: str, payload: Dict[str, Any]):
    evt = {
        'type': event_type,
        'ts': time.time(),
        'payload': payload
    }
    with _sub_lock:
        # Copy to avoid holding lock while putting (queue.put is thread-safe)
        subs = list(_subscribers)
    for q in subs:
        try:
            q.put(evt, block=False)
        except Exception:
            pass

def subscribe_queue() -> Queue:
    q: Queue = Queue(maxsize=1000)
    with _sub_lock:
        _subscribers.append(q)
    return q

def unsubscribe_queue(q: Queue):
    with _sub_lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass

def sse_stream_generator(heartbeat_interval: float = 15.0):
    """Yield SSE formatted lines for a single subscriber until disconnect."""
    q = subscribe_queue()
    last_sent = time.time()
    try:
        while True:
            try:
                evt = q.get(timeout=1.0)
                import json
                data = json.dumps(evt)
                yield f"data: {data}\n\n"
                last_sent = time.time()
            except Empty:
                # Heartbeat
                if time.time() - last_sent >= heartbeat_interval:
                    yield "event: heartbeat\ndata: {}\n\n"
                    last_sent = time.time()
    except GeneratorExit:  # client disconnected
        pass
    finally:
        unsubscribe_queue(q)
