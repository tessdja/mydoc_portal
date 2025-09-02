import os
import json
import secrets  # for default session secret

from starlette.middleware.sessions import SessionMiddleware
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Body, Depends
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from utils.document_ops import FastAPIFileAdapter
from utils.db import safe_select

from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparator,
    ChatIngestor,
)
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG
from utils.document_ops import FastAPIFileAdapter,read_pdf_via_handler
from logger import GLOBAL_LOGGER as log

# -----------FOLLOWING IMPORTS SUPPORT NL TO SQL ROUTE---------
from utils.schema_catalog import build_catalog, catalog_as_prompt_text
from utils.nl_to_sql import generate_and_validate
from utils.db import safe_select
from utils.intent import classify_intent

from prompt.prompt_library import PROMPT_REGISTRY
from utils.model_loader import ModelLoader

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")  # <--- keep consistent with save_local()

app = FastAPI(title="Document Portal API", version="0.1")

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session cookies for simple login
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", secrets.token_urlsafe(32)),
    same_site="lax",
    https_only=False,  # set True if you are strictly on HTTPS
)

def get_current_user(request: Request) -> str:
    user = request.session.get("user")
    if not user:
        # For API routes we prefer a 401 so the frontend can react
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    # capture ?next=... if provided
    next_target = request.query_params.get("next", "/")
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": None, "next": next_target}
    )

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request,
                username: str = Form(...),
                password: str = Form(...),
                next: str | None = Form(default="/")):
    valid_user = os.getenv("APP_USER", "admin")
    valid_pass = os.getenv("APP_PASS", "changeme")

    if username == valid_user and password == valid_pass:
        request.session["user"] = username
        return RedirectResponse(next or "/", status_code=303)

    # Bad creds – re-render form with an error
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid username or password."},
        status_code=401
    )

@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/login", status_code=302)
    log.info("Serving UI homepage.")
    resp = templates.TemplateResponse("index.html", {
        "request": request,
        "user": request.session.get("user")
    })
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/health")
def health() -> Dict[str, str]:
    log.info("Health check passed.")
    return {"status": "ok", "service": "document-portal"}

# ---------- ANALYZE ----------
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...),
                            user: str = Depends(get_current_user)) -> Any:
    try:
        log.info(f"Received file for analysis: {file.filename}")
        dh = DocHandler()
        
        #saved_path = dh.save_pdf(FastAPIFileAdapter(file))
        #text = read_pdf_via_handler(dh, saved_path)
        saved_path = dh.save_any(FastAPIFileAdapter(file))
        text = dh.read_any(saved_path) 

        if not text or not text.strip():
            # Surface a clear error instead of letting the JSON parser explode
            raise HTTPException(status_code=400, detail="No text extracted from the file. If this is a CSV," \
            "ensure the loader creates a text preview")

        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        log.info("Document analysis complete.")
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Error during document analysis")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

# ---------- COMPARE ----------
def _tabular_to_page_changes_rows(result: dict) -> list[dict[str, str]]:
    """Format tabular diff into your current UI's 'Page'/'Changes' rows."""
    rows = []

    # Added / Removed rows summary
    if result.get("added_rows"):
        rows.append({"Page": "Added rows", "Changes": str(len(result["added_rows"]))})
    if result.get("removed_rows"):
        rows.append({"Page": "Removed rows", "Changes": str(len(result["removed_rows"]))})

    # Per-column summaries (e.g., 'Private -> Self-employed (42); ...')
    for col_block in result.get("summaries", []):
        col = col_block.get("column", "")
        changes = col_block.get("changes", [])
        if changes:
            summary_text = "; ".join(f"{c['change']} ({c['count']})" for c in changes)
        else:
            summary_text = "NO CHANGE"
        rows.append({"Page": col, "Changes": summary_text})

    # If everything was empty, emit a single NO CHANGE row
    if not rows:
        rows.append({"Page": "All", "Changes": "NO CHANGE"})
    return rows


@app.post("/compare")
async def compare_documents(reference: UploadFile = File(...), actual: UploadFile = File(...),
                            user: str = Depends(get_current_user)) -> Any:
    try:
        log.info(f"Comparing files: {reference.filename} vs {actual.filename}")

        dc = DocumentComparator()
        ref_path, act_path = dc.save_uploaded_files(
            FastAPIFileAdapter(reference), FastAPIFileAdapter(actual)
        )

        combined = dc.combine_pair(ref_path, act_path)

        # ----------  Excel/CSV path: 'combined' is a dict  ----------
        if isinstance(combined, dict):
            # Keep existing UI working: provide 'rows' (Page/Changes) AND the full structured diff.
            page_rows = _tabular_to_page_changes_rows(combined)
            log.info("Tabular diff completed",
                        key_columns=combined.get("key_columns"),
                        cols=len(combined.get("columns_compared", [])),
                        session_id=dc.session_id)
            return {
                "rows": page_rows,          # your current table will render this
                "tabular": combined,        # full structured diff for richer UIs later
                "session_id": dc.session_id
            }

        # ----------  Non-tabular path: 'combined' is text for the LLM  ----------
        text = combined if isinstance(combined, str) else str(combined)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from one or both files.")

        comp = DocumentComparatorLLM()
        df = comp.compare_documents(text)

        log.info("Document comparison completed.", session_id=dc.session_id)
        return {"rows": df.to_dict(orient="records"), "session_id": dc.session_id}

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

# ---------- CHAT: INDEX ----------
@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    user: str = Depends(get_current_user),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
) -> Any:
    try:
        log.info(f"Indexing chat session. Session ID: {session_id}, Files: {[f.filename for f in files]}")
        wrapped = [FastAPIFileAdapter(f) for f in files]
        # this is my main class for storing a data into VDB
        # created a object of ChatIngestor
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        # NOTE: ensure your ChatIngestor saves with index_name="index" or FAISS_INDEX_NAME
        # e.g., if it calls FAISS.save_local(dir, index_name=FAISS_INDEX_NAME)
        ci.built_retriver(  # if your method name is actually build_retriever, fix it there as well
            wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k
        )
        log.info(f"Index created successfully for session: {ci.session_id}")
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat index building failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

# ---------- CHAT: QUERY ----------
@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    user: str = Depends(get_current_user),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
    return_contexts: bool = Form(False), 
    mode: str = Form("auto"),            # NEW: 'auto' | 'db' | 'docs' | 'hybrid'
    schema: str = Form("docportal"),     # NEW: DB schema name
    refresh_schema: bool = Form(False),  # NEW: force catalog refresh
) -> Any:
    """
    Returns a uniform envelope:
    { "mode": "db|docs|hybrid",
        "answer": str,
        "sql": str|None,
        "rows": list|None,
        "sources": list|None }
    """
    try:
        # pick intent
        intent = mode if mode in {"db", "docs", "hybrid"} else classify_intent(question)

        # ---- DOCS path (RAG) ----
        doc_answer, sources = None, None
        if intent in ("docs", "hybrid"):
            if use_session_dirs and not session_id:
                raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")
            index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE
            if not os.path.isdir(index_dir):
                raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")
            rag = ConversationalRAG(session_id=session_id)
            rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)

            if return_contexts:
                doc_answer, sources = rag.invoke_with_contexts(question, chat_history=[], top_k=k)
            else:
                doc_answer, sources = rag.invoke(question, chat_history=[]), []

        # ---- DB path (NL→SQL) ----
        sql, rows = None, None
        if intent in ("db", "hybrid"):
            cat = build_catalog(schema=schema, table_whitelist=None,
                                ttl_seconds=0 if refresh_schema else 3600)
            cat_text = catalog_as_prompt_text(cat)
            examples = [
                ("Count rows in v2", "SELECT COUNT(*) AS n FROM docportal.stroke_patients_v2"),
                ("work_type counts for Rural",
                "SELECT work_type, COUNT(*) AS cnt FROM docportal.stroke_patients_v2 "
                "WHERE residence_type='Rural' GROUP BY work_type ORDER BY cnt DESC"),
                ("Ids that changed work_type from Private to Self-employed between v1 and v2",
                "SELECT v1.id FROM docportal.stroke_patients_v1 v1 "
                "JOIN docportal.stroke_patients_v2 v2 USING (id) "
                "WHERE v1.work_type='Private' AND v2.work_type='Self-employed'"),
            ]
            sql = generate_and_validate(
                question, schema_text=cat_text, examples=examples,
                max_rows=int(os.getenv("MYSQL_MAX_ROWS", "1000")),
            )
            rows = safe_select(sql)

        # ---- compose response ----
        if intent == "db":
            return {"mode": "db", "answer": f"Query executed for: {question}", 
                    "sql": sql, "rows": rows, "sources": []}

        # ---- compose response ----
        if intent == "db":
            return {"mode": "db", "answer": f"Query executed for: {question}", 
                    "sql": sql, "rows": rows, "sources": []}

        if intent == "docs":
            payload = {"mode": "docs", "answer": doc_answer, 
                        "sql": None, "rows": None}
            if return_contexts:
                payload["sources"] = sources
            return payload

        # hybrid fusion
        llm = ModelLoader().load_llm()
        prompt = PROMPT_REGISTRY["hybrid_answer"]
        messages = prompt.format_messages(
            question=question,
            db_rows_json=json.dumps(rows[:50] if rows else [], ensure_ascii=False, default=str),
            snippets="\n\n".join(sources or []),
        )
        fusion = llm.invoke(messages)
        hybrid_answer = getattr(fusion, "content", fusion)

        payload = {"mode": "hybrid", "answer": hybrid_answer, "sql": sql, "rows": rows}
        if return_contexts:
            payload["sources"] = sources
        return payload


    except HTTPException:
        raise
    except Exception as e:
        log.exception("chat_query failed")
        raise HTTPException(status_code=500, detail=f"chat_query failed: {e}")

    
# ---TEST MYSQL CONNECTION ----------
@app.post("/db/query")
async def db_query(payload: dict):
    try:
        rows = safe_select(payload["sql"], tuple(payload.get("params", [])))
        return {"rows": rows}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
# ----- NL TO SQL with schema catalog and guardrails ------------
@app.post("/db/nl")
async def db_nl(payload: dict = Body(...)):
    try:
        question = (payload.get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Missing 'question'")

        schema = payload.get("schema", "docportal")
        refresh = bool(payload.get("refresh", False))

        # 1) catalog
        cat = build_catalog(schema=schema, table_whitelist=None,
                            ttl_seconds=0 if refresh else 3600)
        cat_text = catalog_as_prompt_text(cat)

        # 2) few-shots
        examples = [
            ("Count rows in v2", "SELECT COUNT(*) AS n FROM docportal.stroke_patients_v2"),
            ("work_type counts for Rural",
                "SELECT work_type, COUNT(*) AS cnt FROM docportal.stroke_patients_v2 "
                "WHERE residence_type='Rural' GROUP BY work_type ORDER BY cnt DESC"),
            ("Ids that changed work_type from Private to Self-employed between v1 and v2",
                "SELECT v1.id FROM docportal.stroke_patients_v1 v1 "
                "JOIN docportal.stroke_patients_v2 v2 USING (id) "
                "WHERE v1.work_type='Private' AND v2.work_type='Self-employed'"),
        ]

        # 3) generate + validate
        max_rows = int(os.getenv("MYSQL_MAX_ROWS", "1000"))
        safe_sql = generate_and_validate(
            question, schema_text=cat_text, examples=examples, max_rows=max_rows
        )
        log.info("NL2SQL", sql=safe_sql)

        # 4) execute
        rows = safe_select(safe_sql)
        return {"sql": safe_sql, "rows": rows}

    except HTTPException:
        raise
    except Exception as e:
        log.exception("db_nl failed")   # <-- now you’ll see the real error in the console
        raise HTTPException(status_code=400, detail=f"db_nl failed: {type(e).__name__}: {e}")




# command for executing the fast api
# uvicorn api.main:app --port 8080 --reload    
#uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload