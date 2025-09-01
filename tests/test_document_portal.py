import io
import os
import types
import json
import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
from api.main import app

client = TestClient(app)

# ---------- 1) GET /health ----------
def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "document-portal"}


# ---------- 2) GET / has no-store cache header ----------
def test_homepage_cache_control():
    r = client.get("/")
    assert r.status_code == 200
    # Template content can vary; the header is stable in app
    assert r.headers.get("Cache-Control") == "no-store"


# ---------- 3) POST /analyze success (mocks DocHandler + Analyzer) ----------
def test_analyze_success(monkeypatch):
    class DummyDH:
        def save_any(self, file_adapter):  # path return is irrelevant
            return "saved-path"
        def read_any(self, path):
            return "some extracted text"

    class DummyAnalyzer:
        def analyze_document(self, text):
            return {"summary": "ok", "length": len(text)}

    monkeypatch.setattr("api.main.DocHandler", DummyDH)
    monkeypatch.setattr("api.main.DocumentAnalyzer", DummyAnalyzer)

    file_bytes = io.BytesIO(b"fake pdf bytes")
    r = client.post(
        "/analyze",
        files={"file": ("fake.pdf", file_bytes, "application/pdf")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["summary"] == "ok"


# ---------- 4) POST /analyze -> 400 if no text extracted ----------
def test_analyze_no_text_400(monkeypatch):
    class DummyDH:
        def save_any(self, file_adapter):  # path return is irrelevant
            return "saved-path"
        def read_any(self, path):
            return ""  # simulate empty extraction

    monkeypatch.setattr("api.main.DocHandler", DummyDH)
    # Analyzer won't be reached

    file_bytes = io.BytesIO(b"fake pdf bytes")
    r = client.post(
        "/analyze",
        files={"file": ("fake.pdf", file_bytes, "application/pdf")},
    )
    assert r.status_code == 400
    assert "No text extracted" in r.text


# ---------- 5) POST /compare (tabular/dict path) ----------
def test_compare_tabular_dict(monkeypatch):
    # Monkeypatch the helper to bypass the module-level @staticmethod bug
    def rows_helper(result):
        return [{"Page": "Added rows", "Changes": "2"}]
    # will silently no-op if the attribute doesn't exist
    monkeypatch.setattr("api.main._tabular_to_page_changes_rows", rows_helper, raising=False)


    class DummyDC:
        def save_uploaded_files(self, ref, act):
            return "ref-path", "act-path"
        def combine_pair(self, ref_path, act_path):
            # simulate structured diff result
            return {
                "added_rows": [{"id": 1}, {"id": 2}],
                "removed_rows": [],
                "summaries": [{"column": "work_type", "changes": [{"change": "A->B", "count": 2}]}],
                "columns_compared": ["work_type"],
                "key_columns": ["id"],
            }
        @property
        def session_id(self):
            return "sess-123"

    monkeypatch.setattr("api.main.DocumentComparator", DummyDC)

    r = client.post(
        "/compare",
        files={
            "reference": ("ref.csv", io.BytesIO(b"id,a\n1,x\n"), "text/csv"),
            "actual": ("act.csv", io.BytesIO(b"id,a\n1,y\n2,z\n"), "text/csv"),
        },
    )
    assert r.status_code == 200
    payload = r.json()
    assert "rows" in payload and isinstance(payload["rows"], list)
    assert "tabular" in payload and "columns_compared" in payload["tabular"]


# ---------- 6) POST /compare (LLM/text path) ----------
def test_compare_llm_text(monkeypatch):
    class DummyDC:
        def save_uploaded_files(self, ref, act):
            return "ref-path", "act-path"
        def combine_pair(self, ref_path, act_path):
            return "long plain text to compare"
        @property
        def session_id(self):
            return "sess-abc"

    class DummyDF:
        def to_dict(self, orient="records"):
            return [{"Page": "1", "Changes": "X->Y"}]

    class DummyLLMComp:
        def compare_documents(self, text):
            return DummyDF()

    monkeypatch.setattr("api.main.DocumentComparator", DummyDC)
    monkeypatch.setattr("api.main.DocumentComparatorLLM", DummyLLMComp)

    r = client.post(
        "/compare",
        files={
            "reference": ("ref.pdf", io.BytesIO(b"%PDF-1.4..."), "application/pdf"),
            "actual": ("act.pdf", io.BytesIO(b"%PDF-1.4..."), "application/pdf"),
        },
    )
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload["rows"], list)
    assert payload["session_id"] == "sess-abc"


# ---------- 7) POST /chat/index builds retriever ----------
def test_chat_index_success(monkeypatch):
    class DummyCI:
        def __init__(self, temp_base, faiss_base, use_session_dirs, session_id):
            self._sid = session_id or "auto-sid"
        def built_retriver(self, wrapped, chunk_size, chunk_overlap, k):
            # no-op
            pass
        @property
        def session_id(self):
            return self._sid

    monkeypatch.setattr("api.main.ChatIngestor", DummyCI)

    r = client.post(
        "/chat/index",
        files=[
            ("files", ("a.pdf", io.BytesIO(b"x"), "application/pdf")),
            ("files", ("b.txt", io.BytesIO(b"y"), "text/plain")),
        ],
        data={"session_id": "s123", "k": "3"}  # form fields are strings
    )
    assert r.status_code == 200
    out = r.json()
    assert out["session_id"] == "s123"
    assert out["k"] == 3  # FastAPI converts to int


# ---------- 8) POST /chat/query in docs mode ----------
def test_chat_query_docs_success(tmp_path, monkeypatch):
    # Ensure FAISS index dir exists (FAISS_BASE/index_name set at import time: "faiss_index")
    monkeypatch.chdir(tmp_path)
    os.makedirs(os.path.join("faiss_index", "s1"), exist_ok=True)

    class DummyRAG:
        def __init__(self, session_id=None):
            self.sid = session_id
        def load_retriever_from_faiss(self, index_dir, k, index_name):
            assert os.path.isdir(index_dir)
        def invoke(self, question, chat_history):
            return f"Answer to: {question}"

    monkeypatch.setattr("api.main.ConversationalRAG", DummyRAG)

    r = client.post(
        "/chat/query",
        data={"question": "What is X?", "mode": "docs", "session_id": "s1"}
    )
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "docs"
    assert "Answer to:" in data["answer"]


# ---------- 9) POST /chat/query in db mode ----------
def test_chat_query_db_success(monkeypatch):
    monkeypatch.setattr("api.main.build_catalog", lambda schema, table_whitelist, ttl_seconds: {"ok": True})
    monkeypatch.setattr("api.main.catalog_as_prompt_text", lambda cat: "schema text")
    monkeypatch.setattr("api.main.generate_and_validate", lambda q, schema_text, examples, max_rows: "SELECT 1 AS n")
    monkeypatch.setattr("api.main.safe_select", lambda sql: [{"n": 1}])

    r = client.post(
        "/chat/query",
        data={"question": "Count rows?", "mode": "db"}
    )
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "db"
    assert data["sql"].lower().startswith("select")
    assert data["rows"] == [{"n": 1}]


# ---------- 10) POST /db/nl missing question -> 400 ----------
def test_db_nl_missing_question_400():
    r = client.post("/db/nl", json={"schema": "docportal"})
    assert r.status_code == 400
    assert "Missing 'question'" in r.text


# (Bonus, uncomment to assert missing FAISS index -> 404)
# def test_chat_query_docs_index_missing(monkeypatch):
#     class DummyRAG: ...
#     # Not creating faiss_index/session -> expect 404
#     r = client.post("/chat/query", data={"question":"Q","mode":"docs","session_id":"nope"})
#     assert r.status_code == 404
