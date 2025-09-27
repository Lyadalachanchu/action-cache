# cachedb/db.py
import sqlite3
from pathlib import Path
from typing import Optional

from .config import DB_PATH

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(schema_path: Optional[str] = None):
    if schema_path is None:
        schema_path = str(Path(__file__).with_name("schema.sql"))
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = f.read()
    conn = get_connection()
    with conn:
        conn.executescript(schema)
    conn.close()
