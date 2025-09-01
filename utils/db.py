# db.py
import os, re, mysql.connector
from dotenv import load_dotenv
from mysql.connector import connect
from logger import GLOBAL_LOGGER as log

# Load .env for local runs (do this here so /db/query has values)
if os.getenv("ENV", "local").lower() != "production":
    load_dotenv()
    log.info("DB: .env loaded")
    # print(os.getenv("MYSQL_USER"))
    # print(os.getenv("MYSQL_PASSWORD"))

_BAD = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|RENAME|GRANT|REVOKE)\b", re.I)

def get_conn():
    return connect(
        host=os.getenv("MYSQL_HOST","localhost"),
        port=int(os.getenv("MYSQL_PORT","3306")),
        database=os.getenv("MYSQL_DB","docportal"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        autocommit=True,
        connection_timeout=10,
    )

def safe_select(sql: str, params: tuple | None = None, max_rows: int = 1000):
    s = sql.strip().rstrip(";")
    if _BAD.search(s) or not s.lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")
    if " limit " not in s.lower():
        s = f"{s} LIMIT {max_rows}"
    with get_conn() as cn, cn.cursor(dictionary=True) as cur:
        cur.execute(s, params or ())
        return cur.fetchall()
