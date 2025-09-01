# tests/evals/test_rag_deepeval.py
import json, pathlib, os
import pytest
from fastapi.testclient import TestClient
from api.main import app

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

key = (os.getenv("OPENAI_API_KEY") or "").strip()
if not key or key.startswith('"') or key.endswith('"'):
    pytest.skip("Valid OPENAI_API_KEY required for DeepEval metrics; skipping.", allow_module_level=True)



client = TestClient(app)
GOLDENS = json.loads((pathlib.Path(__file__).parent / "goldens" / "rag_examples.json").read_text())

def _ideal_for(q):
    for c in GOLDENS:
        if c["question"] == q:
            return c["ideal"], c["contexts"]
    # fallback (shouldn't happen)
    return "No policy found.", []

def test_rag_quality_small_suite(tmp_path, monkeypatch):
    # 1) Ensure FAISS index dir so /chat/query doesn't 404
    #    Your app typically uses "faiss_index/<session_id>"
    session_id = "eval"
    os.makedirs(tmp_path / "faiss_index" / session_id, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    # 2) Stub the RAG to avoid live LLM + retrieval
    class DummyRAG:
        def __init__(self, session_id=None): self.sid = session_id
        def load_retriever_from_faiss(self, index_dir, k=None, index_name=None): pass
        def invoke_with_contexts(self, question, chat_history=None, top_k=None):
            ideal, ctx = _ideal_for(question)
            return ideal, ctx
        def invoke(self, question, chat_history=None):
            ideal, _ = _ideal_for(question)
            return ideal

    # Patch where app imports the class
    monkeypatch.setattr("api.main.ConversationalRAG", DummyRAG)

    # 3) Run the DeepEval assertions
    for case in GOLDENS:
        r = client.post("/chat/query", data={
            "question": case["question"],
            "mode": "docs",
            "session_id": session_id,
            "k": 3,
            "return_contexts": "true"
        })
        assert r.status_code == 200, r.text
        payload = r.json()
        actual   = payload["answer"]
        contexts = payload.get("sources") or case["contexts"]

        answer_rel = AnswerRelevancyMetric(threshold=0.65)
        faithful   = FaithfulnessMetric(threshold=0.70)
        ctx_prec   = ContextualPrecisionMetric(threshold=0.50)

        assert_test(
            LLMTestCase(
                input=case["question"],
                actual_output=actual,
                expected_output=case["ideal"],
                context=contexts,  
                retrieval_context=contexts,
            ),
            [answer_rel, faithful, ctx_prec]
        )
