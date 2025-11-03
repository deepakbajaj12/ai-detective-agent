# --- Advanced NLP Features ---
# Make spaCy optional and only load when needed to avoid import-time failures.
try:
    import spacy as _spacy  # type: ignore
    _HAS_SPACY = True
except Exception:
    _spacy = None  # type: ignore
    _HAS_SPACY = False

def _load_spacy_model():
    if not _HAS_SPACY:
        return None
    try:
        return _spacy.load('en_core_web_sm')  # type: ignore
    except Exception:
        # Model not available; caller can fallback
        return None

def extract_entities_and_relations(clues):
    """
    Extracts named entities (person, location, date, etc.) and relationships from clues using spaCy.
    If spaCy or the model isn't available, returns empty entities/relations (graceful fallback).
    Returns a list of dicts: {clue, entities, relations}
    """
    nlp = _load_spacy_model()
    results = []
    for clue in clues:
        if nlp is None:
            # Fallback: no NLP, return empty annotations
            results.append({'clue': clue, 'entities': [], 'relations': []})
            continue
        doc = nlp(clue)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        # Simple relation extraction: look for subject-verb-object triples
        relations = []
        for token in doc:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                subj = [w for w in token.lefts if w.dep_ in ('nsubj', 'nsubjpass')]
                obj = [w for w in token.rights if w.dep_ in ('dobj', 'pobj', 'attr')]
                if subj and obj:
                    relations.append((subj[0].text, token.text, obj[0].text))
        results.append({
            'clue': clue,
            'entities': entities,
            'relations': relations
        })
    return results

# --- Automated Clue Extraction from Text/PDFs ---
import pdfplumber

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF file using pdfplumber.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_clues_from_text(text, keywords=None):
    """
    Extracts clue-like sentences from text using spaCy NLP when available.
    Fallback: naive sentence split and keyword filtering.
    Optionally filter by keywords (list of strings).
    """
    nlp = _load_spacy_model()
    clues = []
    if nlp is not None:
        doc = nlp(text)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if keywords:
                if any(kw.lower() in sent_text.lower() for kw in keywords):
                    clues.append(sent_text)
            else:
                # Heuristic: sentences with signal words
                if any(word in sent_text.lower() for word in ['clue', 'found', 'evidence', 'suspect', 'note', 'fingerprint', 'door', 'glass']):
                    clues.append(sent_text)
        return clues
    # Fallback without spaCy: naive split
    import re
    sentences = [s.strip() for s in re.split(r'[\.!?\n]+', text) if s.strip()]
    for sent_text in sentences:
        if keywords:
            if any(kw.lower() in sent_text.lower() for kw in keywords):
                clues.append(sent_text)
        else:
            if any(word in sent_text.lower() for word in ['clue', 'found', 'evidence', 'suspect', 'note', 'fingerprint', 'door', 'glass']):
                clues.append(sent_text)
    return clues
def test_visualize_case_graph():
    clues = ["The door was unlocked.", "A note was found on the table.", "No fingerprints on the glass."]
    suspects = ["Alice", "Bob"]
    events = ["Entry", "Note discovery"]
    relationships = [
        ("The door was unlocked.", "Entry", "related"),
        ("A note was found on the table.", "Note discovery", "related"),
        ("No fingerprints on the glass.", "Alice", "possible suspect"),
        ("Entry", "Bob", "possible suspect"),
        ("Note discovery", "Alice", "possible suspect")
    ]
    visualize_case_graph(clues, suspects, events, relationships)
# --- Interactive Graph Visualization ---
# Requires: pip install networkx matplotlib
import networkx as nx
import matplotlib.pyplot as plt

def visualize_case_graph(clues, suspects, events, relationships):
    """
    Visualizes relationships between clues, suspects, and events as a network graph.
    clues: list of clue strings
    suspects: list of suspect names
    events: list of event descriptions
    relationships: list of tuples (source, target, label)
    """
    G = nx.DiGraph()
    for clue in clues:
        G.add_node(clue, type='clue')
    for suspect in suspects:
        G.add_node(suspect, type='suspect')
    for event in events:
        G.add_node(event, type='event')
    for src, tgt, label in relationships:
        G.add_edge(src, tgt, label=label)

    # Hierarchical layout: suspects (top), clues (middle), events (bottom)
    layers = [suspects, clues, events]
    pos = {}
    y_gap = 2
    for i, layer in enumerate(layers):
        x_gap = 2
        for j, node in enumerate(layer):
            pos[node] = (j * x_gap, -i * y_gap)

    node_colors = []
    for node in G.nodes(data=True):
        if node[1]['type'] == 'clue':
            node_colors.append('skyblue')
        elif node[1]['type'] == 'suspect':
            node_colors.append('salmon')
        else:
            node_colors.append('lightgreen')

    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=1200, font_size=10, arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title('Case Relationships Graph (Hierarchical)')
    plt.tight_layout()
    plt.show()
import pandas as pd
import numpy as np

def clean_clues(clues):
    """
    Cleans and preprocesses a list of clues.
    Removes duplicates, handles missing values, and returns a cleaned DataFrame.
    """
    df = pd.DataFrame({'clue': clues})
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df['clue'] = df['clue'].str.strip()
    return df

# --- SQL Integration ---
# Optional connectors. Import lazily to avoid breaking base app when not installed.
try:
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover
    psycopg2 = None  # type: ignore
try:
    import mysql.connector as mysql_connector  # type: ignore
except Exception:  # pragma: no cover
    mysql_connector = None  # type: ignore

def connect_postgresql(host, dbname, user, password):
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed. Install psycopg2-binary to use PostgreSQL features.")
    return psycopg2.connect(host=host, dbname=dbname, user=user, password=password)

def connect_mysql(host, dbname, user, password):
    if mysql_connector is None:
        raise RuntimeError("mysql-connector-python is not installed. Install it to use MySQL features.")
    return mysql_connector.connect(host=host, database=dbname, user=user, password=password)

def create_clues_table_pg(conn):
    with conn.cursor() as cur:
        cur.execute('''CREATE TABLE IF NOT EXISTS clues (
            id SERIAL PRIMARY KEY,
            clue TEXT NOT NULL
        )''')
        conn.commit()

def create_clues_table_mysql(conn):
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS clues (
        id INT AUTO_INCREMENT PRIMARY KEY,
        clue TEXT NOT NULL
    )''')
    conn.commit()
    cur.close()

def insert_clue_pg(conn, clue):
    with conn.cursor() as cur:
        cur.execute('INSERT INTO clues (clue) VALUES (%s)', (clue,))
        conn.commit()

def insert_clue_mysql(conn, clue):
    cur = conn.cursor()
    cur.execute('INSERT INTO clues (clue) VALUES (%s)', (clue,))
    conn.commit()
    cur.close()

def get_clues_pg(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT clue FROM clues')
        return [row[0] for row in cur.fetchall()]

def get_clues_mysql(conn):
    cur = conn.cursor()
    cur.execute('SELECT clue FROM clues')
    clues = [row[0] for row in cur.fetchall()]
    cur.close()
    return clues

# --- NumPy Statistical Reasoning ---
def clue_length_stats(clues):
    """
    Returns basic statistics (mean, std, min, max) on clue lengths using NumPy.
    """
    lengths = np.array([len(clue) for clue in clues])
    return {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths)
    }
def read_clues(file_path: str) -> list[str]:
    """
    Reads clues from a text file.
    Each line is treated as one clue.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            clues = [line.strip() for line in f if line.strip()]
        return clues
    except FileNotFoundError:
        print(f"⚠️ File not found: {file_path}")
        return []


