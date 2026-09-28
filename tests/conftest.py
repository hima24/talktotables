"""Shared fixtures: a small SYNTHETIC Olist-shaped database built on the fly.

Tests never need the real Kaggle download or an API key.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import make_sample_data  # noqa: E402
from src import config, llm  # noqa: E402
from src.db import Database  # noqa: E402
from src.load_data import load  # noqa: E402
from src.retriever import Retriever  # noqa: E402


@pytest.fixture(scope="session")
def sample_db_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("olist")
    make_sample_data.main(["--out", str(d / "raw"), "--orders", "600"])
    db_path = d / "sample.duckdb"
    load(d / "raw", db_path, check_counts=False)
    return db_path


@pytest.fixture(scope="session")
def db(sample_db_path):
    database = Database(sample_db_path, max_rows=500, timeout_s=5)
    yield database
    database.close()


@pytest.fixture(scope="session")
def retriever(tmp_path_factory):
    return Retriever(backend="bm25", index_path=tmp_path_factory.mktemp("idx") / "rag.duckdb")


@pytest.fixture(scope="session")
def gold_by_question():
    qs = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    return {q["question"]: q["sql"] for q in qs}


@pytest.fixture
def fake_model(monkeypatch):
    """Register a scripted 'fake' model. Set fake_model.replies to a list of strings."""

    class Fake:
        replies: list[str] = []
        calls: list[dict] = []

    fake = Fake()
    fake.replies, fake.calls = [], []

    def provider(spec, system_blocks, messages, max_tokens):
        fake.calls.append({"system": system_blocks, "messages": messages})
        text = fake.replies.pop(0) if fake.replies else "The answer is in the table."
        return llm.LLMResponse(text=text, model="fake", input_tokens=100, output_tokens=20)

    llm.register_provider("fake", provider)
    monkeypatch.setitem(config.MODELS, "fake", {"provider": "fake", "model_id": "fake-1", "label": "Fake",
                                                 "price": (1.0, 5.0), "temperature": 0})
    return fake
