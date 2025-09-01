import os

from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    log.info("Serving UI homepage.")
    resp = templates.TemplateResponse("index.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/health")
def health() -> Dict[str, str]:
    log.info("Health check passed.")
    return {"status": "ok", "service": "document-portal"}

# ---------- ANALYZE ----------
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
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
@staticmethod
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
async def compare_documents(reference: UploadFile = File(...), actual: UploadFile = File(...)) -> Any:
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
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
) -> Any:
    try:
        log.info(f"Received chat query: '{question}' | session: {session_id}")
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE  # type: ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)  # build retriever + chain
        response = rag.invoke(question, chat_history=[])
        log.info("Chat query handled successfully.")

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG"
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    
# ---TEST MYSQL CONNECTION ----------
@app.post("/db/query")
async def db_query(payload: dict):
    try:
        rows = safe_select(payload["sql"], tuple(payload.get("params", [])))
        return {"rows": rows}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# command for executing the fast api
# uvicorn api.main:app --port 8080 --reload    
#uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload