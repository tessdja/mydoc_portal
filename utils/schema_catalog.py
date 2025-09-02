# utils/schema_catalog.py
from __future__ import annotations
import os, json, time
from typing import Dict, List, Optional
from utils.db import get_conn

CACHE_DIR = os.path.join(os.getcwd(), "data", "schema_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def build_catalog(schema: str, table_whitelist: Optional[List[str]] = None,
                    sample_rows: int = 50, ttl_seconds: int = 3600) -> Dict:
    """Return a dict catalog; cached on disk."""
    cache_key = f"{schema}__{','.join(sorted(table_whitelist or [])) or 'all'}.json"
    cache_path = os.path.join(CACHE_DIR, cache_key)

    if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path) < ttl_seconds):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    with get_conn() as cn:
        cur = cn.cursor(dictionary=True)

        # 1) tables
        cur.execute("""
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = %s
            ORDER BY TABLE_NAME
        """, (schema,))
        tables = [r["TABLE_NAME"] for r in cur.fetchall()]
        if table_whitelist:
            tables = [t for t in tables if f"{schema}.{t}" in set(table_whitelist)]

        catalog = {"schema": schema, "tables": {}}

        # 2) columns + types
        cur.execute("""
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, COLUMN_KEY, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s
            ORDER BY TABLE_NAME, ORDINAL_POSITION
        """, (schema,))
        for row in cur.fetchall():
            t = row["TABLE_NAME"]
            if t not in tables: continue
            catalog["tables"].setdefault(t, {"columns": [], "primary_key": [], "samples": {}})
            catalog["tables"][t]["columns"].append({
                "name": row["COLUMN_NAME"],
                "type": row["DATA_TYPE"],
                "nullable": row["IS_NULLABLE"] == "YES"
            })
            if row["COLUMN_KEY"] == "PRI":
                catalog["tables"][t]["primary_key"].append(row["COLUMN_NAME"])

        # 3) sample distinct values for short text/enums (helps LLM)
        for t in tables:
            # heuristics: sample for likely categorical columns
            for col in [c["name"] for c in catalog["tables"][t]["columns"]
                        if c["type"] in ("varchar","enum","text","char")]:
                try:
                    cur.execute(f"SELECT DISTINCT `{col}` AS v FROM `{schema}`.`{t}` WHERE `{col}` IS NOT NULL LIMIT %s",
                                (sample_rows,))
                    catalog["tables"][t]["samples"][col] = [r["v"] for r in cur.fetchall()]
                except Exception:
                    pass  # ignore odd columns

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    return catalog

def catalog_as_prompt_text(cat: Dict) -> str:
    lines = [f"SCHEMA: `{cat['schema']}`"]
    for t, meta in cat["tables"].items():
        cols = ", ".join(f"{c['name']}:{c['type']}" for c in meta["columns"])
        pk = ", ".join(meta.get("primary_key") or [])
        lines.append(f"- TABLE `{t}` (pk: {pk or 'unknown'}) → {cols}")
        # include tiny sample hints
        samples = meta.get("samples") or {}
        for col, vals in samples.items():
            if vals:
                lines.append(f"    samples {col}: {', '.join(map(str, vals[:6]))}")
    return "\n".join(lines)
