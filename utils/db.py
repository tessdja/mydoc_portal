# utils/db.py
import os
import re
from typing import Any, List, Dict

def _get_mysql_connector():
    try:
        import mysql.connector  # lazy import
        return mysql.connector
    except ImportError as e:
        raise RuntimeError(
            "mysql-connector-python is required for DB features. "
            "Install it with: pip install mysql-connector-python"
        ) from e

def safe_select(sql: str, params: dict | None = None) -> List[Dict[str, Any]]:
    mysql = _get_mysql_connector()
    conn = mysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
    )
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

