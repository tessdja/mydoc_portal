# utils/nl_to_sql.py
from __future__ import annotations
import os, re
from typing import List, Tuple
from prompt.prompt_library import PROMPT_REGISTRY
from utils.model_loader import ModelLoader

_SELECT_ONLY = re.compile(r"^\s*select\b", re.I | re.S)
_FORBIDDEN   = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|RENAME|GRANT|REVOKE|;)\b", re.I)
_TABLE_NAME  = re.compile(r"`?([a-zA-Z0-9_]+)`?\.`?([a-zA-Z0-9_]+)`?")

def _extract_sql(text: str) -> str:
    m = re.search(r"```sql\s*(.*?)```", text, re.I | re.S)
    s = (m.group(1) if m else text).strip()
    return s.rstrip(";")

def _whitelist_from_schema(schema_text: str) -> List[str]:
    # Expect lines like "Table: docportal.stroke_patients_v1 (…)"
    pat = re.compile(r"Table:\s+([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)")
    return sorted({f"{db}.{tb}" for db, tb in pat.findall(schema_text)})

def enforce_rules(sql: str, *, table_whitelist: List[str], max_rows: int = 1000) -> str:
    s = sql.strip()
    if not _SELECT_ONLY.match(s) or _FORBIDDEN.search(s):
        raise ValueError("Generated SQL violates safety rules (SELECT-only, single statement).")
    if ";" in s:
        raise ValueError("Multiple statements are not allowed.")
    used = {f"{db}.{tb}" for db, tb in _TABLE_NAME.findall(s)}
    unknown = [u for u in used if u not in set(table_whitelist)]
    if unknown:
        raise ValueError(f"Only whitelisted tables allowed: {', '.join(table_whitelist)}")
    if " limit " not in s.lower():
        s = f"{s} LIMIT {max_rows}"
    return s

def generate_sql(question: str, *, schema_text: str, examples: List[Tuple[str, str]]) -> str:
    """Return raw SQL string (not executed)."""
    fewshot = "\n".join(f"Q: {q}\nA:\n```sql\n{s}\n```" for q, s in examples)
    prompt = PROMPT_REGISTRY["nl2sql"]
    messages = prompt.format_messages(schema_text=schema_text, fewshot=fewshot, question=question)

    llm = ModelLoader().load_llm()
    resp = llm.invoke(messages)
    text = getattr(resp, "content", resp)
    return _extract_sql(text)

def generate_and_validate(question: str, *, schema_text: str, examples: List[Tuple[str, str]], max_rows: int = 1000) -> str:
    sql = generate_sql(question, schema_text=schema_text, examples=examples)
    wl = _whitelist_from_schema(schema_text) or ["docportal.stroke_patients_v1", "docportal.stroke_patients_v2"]
    return enforce_rules(sql, table_whitelist=wl, max_rows=max_rows)
