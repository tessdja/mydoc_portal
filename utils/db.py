# utils/db.py
import os
from typing import Any, Dict, List

def _get_mysql_connector():
    try:
        import mysql.connector  # lazy import so importing utils.db doesn't crash in CI
        return mysql.connector
    except ImportError as e:
        raise RuntimeError(
            "mysql-connector-python is required for DB features. "
            "Install it with: pip install mysql-connector-python"
        ) from e

def get_conn():
    """
    Returns a live MySQL connection using env vars.
    Does NOT connect at import time (lazy import above).
    """
    mysql = _get_mysql_connector()
    return mysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
    )

def safe_select(sql: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """
    Convenience query helper returning list[dict].
    """
    conn = get_conn()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params or {})
        rows = cur.fetchall()
        return rows
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()
