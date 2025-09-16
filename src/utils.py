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
