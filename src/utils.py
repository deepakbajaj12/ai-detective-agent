# --- Automated Clue Extraction from Text/PDFs ---
import pdfplumber
import spacy

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
    Extracts clue-like sentences from text using spaCy NLP.
    Optionally filter by keywords (list of strings).
    """
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    clues = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if keywords:
            if any(kw.lower() in sent_text.lower() for kw in keywords):
                clues.append(sent_text)
        else:
            # Heuristic: sentences with 'clue', 'found', 'evidence', 'suspect', 'note', etc.
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
# Requires: pip install psycopg2-binary (for PostgreSQL) or mysql-connector-python (for MySQL)

import psycopg2
import mysql.connector

def connect_postgresql(host, dbname, user, password):
    return psycopg2.connect(host=host, dbname=dbname, user=user, password=password)

def connect_mysql(host, dbname, user, password):
    return mysql.connector.connect(host=host, database=dbname, user=user, password=password)

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
