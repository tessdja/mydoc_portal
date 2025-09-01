# utils/intent.py
from __future__ import annotations
import re

_DB  = re.compile(r"\b(count|how many|sum|avg|average|top\s+\d+|group\s+by|by\s+\w+|where|between|rural|urban)\b", re.I)
_DOC = re.compile(r"\b(explain|summary|summarise|recommend|why|how|pros|cons|action items?)\b", re.I)

def classify_intent(q: str) -> str:
    """Return 'db', 'docs', 'hybrid' or default."""
    q = (q or "").strip()
    db   = bool(_DB.search(q))
    docs = bool(_DOC.search(q))
    if db and docs:
        return "hybrid"
    if db:
        return "db"
    if docs:
        return "docs"
    # default: lean docs unless clearly metric-like
    return "db" if re.search(r"\b(count|top|how many)\b", q, re.I) else "docs"
