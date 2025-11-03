"""Graph builder for relationship visualization.

Derives nodes and edges:
  Suspect (type=suspect)
  Clue (type=clue)
  Document (type=document)
  Offense (type=offense)

Edges:
  (Suspect)-ALLEGED->(Offense)
  (Suspect)-MENTIONED_IN->(Clue)   (heuristic: suspect name appears in clue text)
  (Clue)-FROM->(Document)          (if clue came from auto-ingested document line) [future enhancement]

Currently clues do not store originating document_id; future schema could add it.
"""
from __future__ import annotations
from typing import Dict, Any
import re
try:
    from src.db import get_conn, list_suspects, list_clues, list_allegations  # type: ignore
except Exception:
    from db import get_conn, list_suspects, list_clues, list_allegations  # type: ignore


def build_graph(case_id: str = 'default') -> Dict[str, Any]:
    nodes = []
    edges = []
    with get_conn() as conn:
        suspects = list_suspects(conn, case_id)
        clues = list_clues(conn, case_id=case_id)
        # Build suspect & offense nodes
        suspect_ids = set()
        offense_nodes = {}
        for s in suspects:
            nodes.append({
                'id': f"suspect:{s['id']}",
                'label': s['name'],
                'type': 'suspect'
            })
            suspect_ids.add(s['id'].lower())
            allegations = list_allegations(conn, s['id'], case_id)
            for a in allegations:
                off_id = f"offense:{a['id']}"
                if off_id not in offense_nodes:
                    offense_nodes[off_id] = {
                        'id': off_id,
                        'label': a['offense'],
                        'type': 'offense',
                        'severity': a.get('severity')
                    }
                edges.append({
                    'source': f"suspect:{s['id']}",
                    'target': off_id,
                    'type': 'ALLEGED'
                })
        nodes.extend(offense_nodes.values())
        # Clue nodes + mention edges
        for c in clues:
            cid = f"clue:{c['id']}"
            nodes.append({
                'id': cid,
                'label': c['text'][:60] + ('...' if len(c['text'])>60 else ''),
                'type': 'clue'
            })
            text_low = c['text'].lower()
            for s in suspects:
                name_low = s['name'].lower()
                if re.search(rf"\b{name_low}\b", text_low):
                    edges.append({
                        'source': f"suspect:{s['id']}",
                        'target': cid,
                        'type': 'MENTIONED_IN'
                    })
    return {'nodes': nodes, 'edges': edges, 'meta': {'case_id': case_id}}


__all__ = ['build_graph']