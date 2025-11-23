"""Graph builder + analytics for relationship visualization.

Base graph derives nodes and edges:
    Suspect (type=suspect)
    Clue (type=clue)
    Offense (type=offense)

Edges:
    (Suspect)-ALLEGED->(Offense)
    (Suspect)-MENTIONED_IN->(Clue)   (heuristic: suspect name appears in clue text)

Analytics augmentations (returned by analyze_graph):
    - degree_centrality (normalized) per node
    - betweenness_centrality per node
    - communities (greedy modularity on undirected projection)
    - anomaly flags based on z-score of degree & betweenness
    - basic counts summary
"""
from __future__ import annotations
from typing import Dict, Any, List
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


def analyze_graph(case_id: str = 'default') -> Dict[str, Any]:
    """Return graph plus analytics metrics and community detection.

    For community detection we treat the graph as undirected and run greedy modularity.
    Anomalies: nodes where degree or betweenness z-score > 2.0 (high) or < -2.0 (low) labeled.
    """
    base = build_graph(case_id)
    nodes = base['nodes']
    edges = base['edges']
    try:
        import networkx as nx  # type: ignore
    except Exception:
        return {**base, 'analytics': {'error': 'networkx not available'}}
    G = nx.Graph()
    for n in nodes:
        G.add_node(n['id'], **{k: v for k, v in n.items() if k != 'id'})
    for e in edges:
        # treat as undirected for analytics; weight can differ by type
        w = 1.0
        et = e.get('type')
        if et == 'ALLEGED':
            w = 1.2
        G.add_edge(e['source'], e['target'], type=et, weight=w)
    # Centralities
    try:
        deg_cent = nx.degree_centrality(G)
    except Exception:
        deg_cent = {}
    try:
        bet_cent = nx.betweenness_centrality(G, normalized=True, k=None)
    except Exception:
        bet_cent = {}
    # Communities (best-effort)
    communities: List[List[str]] = []
    try:
        from networkx.algorithms.community import greedy_modularity_communities  # type: ignore
        comms = greedy_modularity_communities(G)
        for cset in comms:
            communities.append(sorted(list(cset)))
    except Exception:
        communities = []
    # Assemble node metrics and anomaly detection
    import math
    degrees = [G.degree(n) for n in G.nodes()] or [0]
    bet_vals = [bet_cent.get(n, 0.0) for n in G.nodes()] or [0.0]
    def _z(vals: List[float], x: float) -> float:
        if not vals:
            return 0.0
        mean = sum(vals)/len(vals)
        var = sum((v-mean)**2 for v in vals)/len(vals)
        std = math.sqrt(var) or 1.0
        return (x - mean)/std
    node_metrics = {}
    for n in G.nodes():
        d_raw = G.degree(n)
        dc = deg_cent.get(n, 0.0)
        bc = bet_cent.get(n, 0.0)
        dz = _z(degrees, d_raw)
        bz = _z(bet_vals, bc)
        anomaly = None
        if dz > 2.0 or bz > 2.0:
            anomaly = 'high_influence'
        elif dz < -2.0:
            anomaly = 'isolated'
        node_metrics[n] = {
            'degree': d_raw,
            'degree_centrality': dc,
            'betweenness_centrality': bc,
            'degree_z': dz,
            'betweenness_z': bz,
            'anomaly': anomaly
        }
    analytics = {
        'node_metrics': node_metrics,
        'communities': communities,
        'community_count': len(communities),
        'edge_count': len(edges),
        'node_count': len(nodes)
    }
    return {**base, 'analytics': analytics}

__all__ = ['build_graph', 'analyze_graph']