from __future__ import annotations
import os
import sys
import json
import uuid
import hashlib
import shutil

import re
import itertools
import pandas as pd

from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict, Any
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from utils.file_io import generate_session_id, save_uploaded_files
from utils.document_ops import load_documents, concat_for_analysis, concat_for_comparison

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
ALLOWED_ANALYZE_EXTS = {".pdf", ".docx", ".txt", ".pptx", ".md", ".csv", ".xlsx"}

# FAISS Manager (load-or-create)
class FaissManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}} ## this is dict of rows
        
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}} # load it if alrady there
            except Exception:
                self._meta = {"rows": {}} # init the empty one if dones not exists
        

        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None
        
    def _exists(self)-> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()
    
    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def _save_meta(self):
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")
        
        
    def add_documents(self,docs: List[Document]):
        
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents_idempotent().")
        
        new_docs: List[Document] = []
        
        for d in docs:
            
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)
            
        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)
    
    def load_or_create(self,texts:Optional[List[str]]=None, metadatas: Optional[List[dict]] = None):
        ## if we running first time then it will not go in this block
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs
        
        
        if not texts:
            raise DocumentPortalException("No existing FAISS index and no data to create one", sys)
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs
        
        
class ChatIngestor:
    def __init__( self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        try:
            self.model_loader = ModelLoader()
            
            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()
            
            self.temp_base = Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base); self.faiss_base.mkdir(parents=True, exist_ok=True)
            
            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            log.info("ChatIngestor initialized",
                      session_id=self.session_id,
                      temp_dir=str(self.temp_dir),
                      faiss_dir=str(self.faiss_dir),
                      sessionized=self.use_session)
        except Exception as e:
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", e) from e
            
        
    def _resolve_dir(self, base: Path):
        if self.use_session:
            d = base / self.session_id # e.g. "faiss_index/abc123"
            d.mkdir(parents=True, exist_ok=True) # creates dir if not exists
            return d
        return base # fallback: "faiss_index/"
        
    def _split(self, docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks
    
    def _prechunk(self, docs: List[Document]) -> List[Document]:
        out = []
        for d in docs:
            src = (d.metadata or {}).get("source", "").lower()
            text = d.page_content or ""
            # Heuristic boundaries
            if src.endswith(".pptx"):
                parts = re.split(r"\n---\s*Slide\s+\d+\s*---\n", text)[1:] or [text]
            elif src.endswith(".md"):
                parts = re.split(r"(?=^#{1,3}\s)", text, flags=re.MULTILINE) or [text]
            else:
                parts = [text]

            for p in parts:
                if p.strip():
                    out.append(Document(page_content=p.strip(), metadata=d.metadata))
        return out


    def built_retriver( self,
        uploaded_files: Iterable,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,):
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")
            
            # NEW: drop empty/near-empty docs
            docs = [d for d in docs if d and d.page_content and d.page_content.strip()]
            if not docs:
                raise ValueError("Files contained no extractable text")
            
            docs = self._prechunk(docs)  # NEW: pre-chunk by slides/headings
            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            ## FAISS manager very very important class for the docchat
            fm = FaissManager(self.faiss_dir, self.model_loader)
            
            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]
            
            try:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
            except Exception:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
                
            added = fm.add_documents(chunks)
            log.info("FAISS index updated", added=added, index=str(self.faiss_dir))
            
            return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
            
        except Exception as e:
            log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e

            
        
            
class DocHandler:
    """
    PDF save + read (page-wise) for analysis.
    Also PPTX, .xlsx, .csv and .md files save + read (page-wise) for analysis
    """
    def __init__(self, data_dir: Optional[str] = None, session_id: Optional[str] = None):
        self.data_dir = data_dir or os.getenv("DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data", "document_analysis"))
        self.session_id = session_id or generate_session_id("session")
        self.session_path = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        log.info("DocHandler initialized", session_id=self.session_id, session_path=self.session_path)

    def save_pdf(self, uploaded_file) -> str:
        try:
            filename = os.path.basename(uploaded_file.name)
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Invalid file type. Only PDFs are allowed.")
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            log.info("PDF saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            log.error("Failed to save PDF", error=str(e), session_id=self.session_id)
            raise DocumentPortalException(f"Failed to save PDF: {str(e)}", e) from e

    def read_pdf(self, pdf_path: str) -> str:
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_chunks.append(f"\n--- Page {page_num + 1} ---\n{page.get_text()}")  # type: ignore
            text = "\n".join(text_chunks)
            log.info("PDF read successfully", pdf_path=pdf_path, session_id=self.session_id, pages=len(text_chunks))
            return text
        except Exception as e:
            log.error("Failed to read PDF", error=str(e), pdf_path=pdf_path, session_id=self.session_id)
            raise DocumentPortalException(f"Could not process PDF: {pdf_path}", e) from e
        
    def save_any(self, uploaded_file) -> str:
        try:
            filename = os.path.basename(uploaded_file.name)
            ext = Path(filename).suffix.lower()
            if ext not in ALLOWED_ANALYZE_EXTS:
                raise ValueError(f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_ANALYZE_EXTS))}")
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            log.info("File saved successfully", 
                     file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            log.error("Failed to save file", error=str(e), session_id=self.session_id)
            raise DocumentPortalException(f"Failed to save file: {str(e)}", e) from e
        
    def read_any(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return self.read_pdf(path)
        # Defer to multi-format loaders
        docs = load_documents([Path(path)])
        return concat_for_analysis(docs)



class DocumentComparator:
    """
    Save, read & combine PDFs for comparison with session-based versioning.
    """
    # TABULAR_EXTS = {".xlsx", ".csv"}
    
    def __init__(self, base_dir: str = "data/document_compare", session_id: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.session_id = session_id or generate_session_id()
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        log.info("DocumentComparator initialized", session_path=str(self.session_path))

    def save_uploaded_files(self, reference_file, actual_file):
        try:
            saved = []
            for fobj in (reference_file, actual_file):
                filename = os.path.basename(fobj.name)
                ext = Path(filename).suffix.lower()
                if ext not in ALLOWED_ANALYZE_EXTS:
                    raise ValueError(f"Invalid file type: Allowed: {','.join(sorted(ALLOWED_ANALYZE_EXTS))}")
                out = self.session_path / filename
                with open(out, "wb") as f:
                    if hasattr(fobj, "read"):
                        f.write(fobj.read())
                    else:
                        f.write(fobj.getbuffer())
                saved.append(out)

            ref_path, act_path = saved
            log.info("Files saved for comparison",
                     reference=str(ref_path), actual=str(act_path), session=self.session_id)
            return ref_path, act_path
        
        except Exception as e:
            log.error("Error saving files", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error saving files", e) from e

    def read_pdf(self, pdf_path: Path) -> str:
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted: {pdf_path.name}")
                parts = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()  # type: ignore
                    if text.strip():
                        parts.append(f"\n --- Page {page_num + 1} --- \n{text}")
            log.info("PDF read successfully", file=str(pdf_path), pages=len(parts))
            return "\n".join(parts)
        except Exception as e:
            log.error("Error reading PDF", file=str(pdf_path), error=str(e))
            raise DocumentPortalException("Error reading PDF", e) from e

    def combine_documents(self) -> str:
        try:
            doc_parts = []
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower() == ".pdf":
                    content = self.read_pdf(file)
                    doc_parts.append(f"Document: {file.name}\n{content}")
            combined_text = "\n\n".join(doc_parts)
            log.info("Documents combined", count=len(doc_parts), session=self.session_id)
            return combined_text
        except Exception as e:
            log.error("Error combining documents", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error combining documents", e) from e
        
    def combine_pair(self, ref_path: Path, act_path: Path) -> str:
        
        TABULAR_EXTS = {".xlsx", ".csv"}

        try:
            # 0) SHORT-CIRCUIT for tabular files (Excel/CSV) — no LLM needed
            if ref_path.suffix.lower() in TABULAR_EXTS and act_path.suffix.lower() in TABULAR_EXTS:
                result = DocumentComparator._tabular_diff_all(ref_path, act_path)  # auto-detect key + compare all overlapping cols
                log.info("Tabular diff completed",
                        key_columns=result.get("key_columns"),
                        cols=len(result.get("columns_compared", [])),
                        session=self.session_id)
                return result            
            
            # 1) NON-TABULAR path (PDF/DOCX/MD/PPTX...) — build combined text for LLM
            ref_docs = load_documents([ref_path])
            act_docs = load_documents([act_path])

            # debugging
            ref_text = "\n".join([d.page_content for d in ref_docs])
            act_text = "\n".join([d.page_content for d in act_docs])  

            log.info("Compare lengths/hashes",
                    ref_len=len(ref_text), act_len=len(act_text),
                    ref_sha=hashlib.sha256(ref_text.encode("utf-8")).hexdigest()[:12],
                    act_sha=hashlib.sha256(act_text.encode("utf-8")).hexdigest()[:12],
                    session=self.session_id)

            if not ref_docs or not act_docs:
                raise ValueError("No text extracted from one or both files.")
            #combined_text = concat_for_comparison(ref_docs, act_docs)
            combined_text = DocumentComparator._concat_with_pages(ref_docs, act_docs, ref_path, act_path)
            log.info("Documents combined for comparison",
                     reference=str(ref_path), actual=str(act_path), session=self.session_id)
            return combined_text
        except Exception as e:
            log.error("Error combining documents", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error combining documents", e) from e
        
    @staticmethod
    def _pseudo_pageize(text: str, *, prefer_headings: bool = True, target_chars: int = 1400) -> str:
        """
        Turns linear text into page-like chunks with markers so the LLM
        can reason 'per page' for non-PDF sources.
        """    
        # 1) Try to split the headings/questions first (works well for your ML Q/A docs)
        if prefer_headings:
            # Heuristics: lines starting with 'Q', 'What', 'Why', 'How', 'A\d+:' etc.
            blocks: List[str] = []
            cur: List[str] = []
            heading_re = re.compile(r'^\s*((Q\d*:?)|(What|Why|How)\b|A\d+\s*:)', re.IGNORECASE)
            for line in text.splitlines():
                if heading_re.match(line) and cur:
                    blocks.append("\n".join(cur).strip())
                    cur = [line]
                else:
                    cur.append(line)
            if cur:
                blocks.append("\n".join(cur).strip())
        else:
            blocks = []

        # 2) If headings didn’t help, fall back to char-length chunking
        if not blocks or sum(len(b) for b in blocks) < len(text) * 0.6:
            blocks = []
            i = 0
            while i < len(text):
                blocks.append(text[i:i+target_chars].strip())
                i += target_chars

        # 3) Prefix with explicit page markers
        pages = []
        for idx, blk in enumerate(blocks, start=1):
            if blk:
                pages.append(f"--- Page {idx} ---\n{blk}")
        return "\n\n".join(pages) if pages else text

    @staticmethod
    def _concat_with_pages(ref_docs: List[Document], act_docs: List[Document],
                       ref_path: Path, act_path: Path) -> str:
        # PDFs keep their real page markers via your PDF pipeline.
        if ref_path.suffix.lower() != ".pdf":
            ref_text = "\n".join(d.page_content for d in ref_docs)
            ref_text = DocumentComparator._pseudo_pageize(ref_text)
            ref_docs = [Document(page_content=ref_text, metadata={"source": str(ref_path)})]
        if act_path.suffix.lower() != ".pdf":
            act_text = "\n".join(d.page_content for d in act_docs)
            act_text = DocumentComparator._pseudo_pageize(act_text)
            act_docs = [Document(page_content=act_text, metadata={"source": str(act_path)})]
        return concat_for_comparison(ref_docs, act_docs)  # keeps <<REFERENCE/ACTUAL>> framing

    # below code is for excel/csv comparison

    @staticmethod
    def _read_tabular(path: Path) -> pd.DataFrame:
        ext = path.suffix.lower()
        if ext == ".xlsx":
            return pd.read_excel(path, dtype=str)
        if ext == ".csv":
            return pd.read_csv(path, dtype=str, low_memory=False)
        raise ValueError(f"Unsupported tabular file: {path.name}")

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        # Make comparisons stable: all strings, normalized whitespace, NA -> ""
        out = df.copy()
        for c in out.columns:
            out[c] = (
                out[c]
                .astype(str)
                .str.replace("\u00A0", " ", regex=False)  # non-breaking space
                .str.replace(r"[ \t]+", " ", regex=True)  # collapse spaces
                .str.replace(r"[ \t]+\n", "\n", regex=True)  # trim right spaces
                .str.replace("\r\n", "\n", regex=False)
                .fillna("")
            )
        return out

    @staticmethod
    def _case_insensitive_get(df: pd.DataFrame, name: str) -> Optional[str]:
        for c in df.columns:
            if c.lower() == name.lower():
                return c
        return None

    @staticmethod
    def _is_unique(series: pd.Series) -> bool:
        # treat empty string as a value; require strict uniqueness
        return series.nunique(dropna=False) == len(series)

    @staticmethod
    def _pick_key_columns(df_ref: pd.DataFrame, df_act: pd.DataFrame, max_combo: int = 2) -> List[str]:
        shared_cols = [c for c in df_ref.columns if c in df_act.columns]
        if not shared_cols:
            raise ValueError("No overlapping columns between files.")

        # 1) Prefer 'id'
        id_ref = DocumentComparator._case_insensitive_get(df_ref, "id")
        id_act = DocumentComparator._case_insensitive_get(df_act, "id")
        if id_ref and id_act and id_ref == id_act:
            if DocumentComparator._is_unique(df_ref[id_ref]) and DocumentComparator._is_unique(df_act[id_act]):
                return [id_ref]

        # 2) Any single unique column
        for c in shared_cols:
            if DocumentComparator._is_unique(df_ref[c]) and DocumentComparator._is_unique(df_act[c]):
                return [c]

        # 3) Try composite keys (pairs)
        # To avoid combinatorial blowup, cap candidates to the first 10 shared columns
        cand = shared_cols[:10]
        for r in range(2, min(max_combo, len(cand)) + 1):
            for cols in itertools.combinations(cand, r):
                if (
                    df_ref[list(cols)].drop_duplicates().shape[0] == len(df_ref)
                    and df_act[list(cols)].drop_duplicates().shape[0] == len(df_act)
                ):
                    return list(cols)

        raise ValueError("Could not auto-detect a unique key. Specify one explicitly.")

    @staticmethod
    def _tabular_diff_all(
        ref_path: Path,
        act_path: Path,
        key_cols: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
    ) -> Dict:
        """Compute deterministic cell-level diffs across ALL overlapping columns."""
        df_ref = DocumentComparator._normalize_df(DocumentComparator._read_tabular(ref_path))
        df_act = DocumentComparator._normalize_df(DocumentComparator._read_tabular(act_path))

        # Pick keys if not given
        if key_cols is None:
            key_cols = DocumentComparator._pick_key_columns(df_ref, df_act)
        else:
            # validate provided key(s)
            for k in key_cols:
                if k not in df_ref.columns or k not in df_act.columns:
                    raise ValueError(f"Key '{k}' not present in both files.")
            if df_ref.drop_duplicates(key_cols).shape[0] != len(df_ref) or df_act.drop_duplicates(key_cols).shape[0] != len(df_act):
                raise ValueError(f"Provided key columns {key_cols} are not unique in both files.")

        # Determine which value columns to compare
        shared_cols = sorted(set(df_ref.columns).intersection(df_act.columns))
        compare_cols = [c for c in shared_cols if c not in key_cols]
        if columns is not None:
            # filter to requested columns (still must be overlapping)
            columns = [c for c in columns if c in compare_cols]
            compare_cols = columns

        # Identify added/removed rows
        ref_keys = df_ref[key_cols].copy()
        ref_keys["__in_ref__"] = True
        act_keys = df_act[key_cols].copy()
        act_keys["__in_act__"] = True
        presence = ref_keys.merge(act_keys, on=key_cols, how="outer")
        added = presence[presence["__in_ref__"].isna()].drop(columns=["__in_ref__", "__in_act__"]).copy()
        removed = presence[presence["__in_act__"].isna()].drop(columns=["__in_ref__", "__in_act__"]).copy()

        # Compare overlapping rows
        merged = df_ref[key_cols + compare_cols].merge(
            df_act[key_cols + compare_cols],
            on=key_cols,
            how="inner",
            suffixes=("_from", "_to"),
        )

        # Cell-level changes
        change_rows: List[Dict] = []
        for col in compare_cols:
            fcol, tcol = f"{col}_from", f"{col}_to"
            diff_mask = merged[fcol] != merged[tcol]
            if diff_mask.any():
                tmp = merged.loc[diff_mask, key_cols + [fcol, tcol]].copy()
                for _, row in tmp.iterrows():
                    key_obj = {k: row[k] for k in key_cols}
                    change_rows.append({
                        "key": key_obj if len(key_cols) > 1 else row[key_cols[0]],
                        "column": col,
                        "from": row[fcol],
                        "to": row[tcol],
                    })

        # Summary per column / change
        if change_rows:
            df_changes = pd.DataFrame(change_rows)
            df_changes["change"] = df_changes["from"] + " -> " + df_changes["to"]
            summaries = (
                df_changes.groupby(["column", "change"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .sort_values(["column", "count"], ascending=[True, False])
            )
            summaries_by_col = []
            for col, sub in summaries.groupby("column"):
                summaries_by_col.append({
                    "column": col,
                    "changes": sub[["change", "count"]].to_dict(orient="records")
                })
            rows_out = df_changes.drop(columns=["change"]).to_dict(orient="records")
        else:
            summaries_by_col = []
            rows_out = []

        return {
            "key_columns": key_cols,
            "columns_compared": compare_cols,
            "added_rows": added.to_dict(orient="records"),     # rows present only in ACTUAL
            "removed_rows": removed.to_dict(orient="records"), # rows present only in REFERENCE
            "summaries": summaries_by_col,                     # per-column change counts
            "rows": rows_out,                                  # cell-level diffs
        }



    def clean_old_sessions(self, keep_latest: int = 3):
        try:
            sessions = sorted([f for f in self.base_dir.iterdir() if f.is_dir()], reverse=True)
            for folder in sessions[keep_latest:]:
                shutil.rmtree(folder, ignore_errors=True)
                log.info("Old session folder deleted", path=str(folder))
        except Exception as e:
            log.error("Error cleaning old sessions", error=str(e))
            raise DocumentPortalException("Error cleaning old sessions", e) from e

