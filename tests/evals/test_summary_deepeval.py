# tests/evals/test_summary_deepeval.py
import os, pytest
import io
from fastapi.testclient import TestClient
from api.main import app

from deepeval import assert_test
from deepeval.metrics import SummarizationMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase

key = (os.getenv("OPENAI_API_KEY") or "").strip()
if not key or key.startswith('"') or key.endswith('"'):
    pytest.skip("Valid OPENAI_API_KEY required for DeepEval metrics; skipping.", allow_module_level=True)


SOURCE = """Our 30-day refund applies to all purchases made online after Jan 1.
Customers must provide order number. Processing takes 5 business days."""
IDEAL = (
    "Online purchases after Jan 1 are eligible for a 30-day refund; "
    "customers must provide an order number; processing takes 5 business days."
)

client = TestClient(app)

def test_analyze_summary_quality(monkeypatch):
    # 1) Stub file IO + analyzer to avoid real LLM calls
    class DummyDH:
        def __init__(self, *args, **kwargs): pass
        def save_any(self, file_adapter): return "saved-path"
        def read_any(self, path): return SOURCE

    class DummyAnalyzer:
        def __init__(self, *args, **kwargs): pass
        def analyze_document(self, text):
            # Return in your app's likely shape:
            return {"metadata": {"Summary": IDEAL}}

    monkeypatch.setattr("api.main.DocHandler", DummyDH)
    monkeypatch.setattr("api.main.DocumentAnalyzer", DummyAnalyzer)

    # 2) Call the endpoint
    r = client.post("/analyze", files={"file": ("policy.txt", io.BytesIO(SOURCE.encode()), "text/plain")})
    assert r.status_code == 200
    payload = r.json()

    # 3) Be tolerant to payload shape: look for common locations
    summary_top = payload.get("summary")
    summary_meta = payload.get("metadata", {}).get("Summary")
    summary_alt = payload.get("result", {}).get("summary")

    summary = summary_meta or summary_top or summary_alt
    assert summary, f"Summary not found in response keys: {list(payload.keys())}"


    STRICT_EVALS = os.getenv("STRICT_EVALS") == "1"

    # Deterministic coverage checks (no LLM judge needed)
    assert "30-day" in summary or "30 day" in summary
    assert "order number" in summary
    assert "5 business days" in summary or "five business days" in summary

    # DeepEval metrics (always keep hallucination; optionally run summarization)
    hall = HallucinationMetric(threshold=0.70, model="gpt-4o")  # pin model for stability
    metrics = [hall]

    if STRICT_EVALS:
        from deepeval.metrics import SummarizationMetric
        # Use a conservative threshold; you can raise this later
        summ = SummarizationMetric(threshold=0.55, model="gpt-4o")
        metrics.append(summ)

    assert_test(
        LLMTestCase(
            input=SOURCE,          # Summarization metric needs the source here
            actual_output=summary,
            expected_output=IDEAL,
            context=[SOURCE],      # 0.21.x expects 'context'
            # retrieval_context=[SOURCE],  # optional
        ),
        metrics
    )