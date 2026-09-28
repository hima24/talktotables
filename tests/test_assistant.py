import numpy as np
import pytest

from eval.compare import results_match
from src import config, llm
from src.assistant import SQLAssistant
from src.retriever import Retriever


@pytest.fixture
def assistant(db, retriever):
    return SQLAssistant(db=db, retriever=retriever)


def test_happy_path_returns_sql_table_and_answer(assistant, fake_model):
    fake_model.replies = ["```sql\nSELECT COUNT(*) AS n FROM orders\n```", "There are 600 orders."]
    res = assistant.ask("How many orders are there?", model="fake")
    assert res.ok and res.attempts == 1
    assert int(res.df.iloc[0, 0]) == 600
    assert res.answer == "There are 600 orders."
    assert res.cost_usd == pytest.approx(2 * (100 * 1.0 + 20 * 5.0) / 1e6)


def test_sql_error_gets_one_self_correction(assistant, fake_model):
    fake_model.replies = [
        "```sql\nSELECT COUNT(*) FROM orders WHERE no_such_column = 1\n```",
        "```sql\nSELECT COUNT(*) FROM orders\n```",
    ]
    res = assistant.ask("How many orders?", model="fake", summarize=False)
    assert res.ok and res.attempts == 2
    # the DuckDB error was sent back to the model
    assert "failed" in fake_model.calls[1]["messages"][-1]["content"]


def test_gives_up_after_retry(assistant, fake_model):
    fake_model.replies = ["```sql\nSELECT nope FROM orders\n```", "```sql\nSELECT still_nope FROM orders\n```"]
    res = assistant.ask("How many orders?", model="fake", summarize=False)
    assert not res.ok and res.attempts == 2 and "SQL error" in res.error


def test_unsafe_sql_is_refused_and_not_retried(assistant, fake_model):
    fake_model.replies = ["```sql\nDROP TABLE orders\n```", "```sql\nSELECT 1 FROM orders\n```"]
    res = assistant.ask("Delete everything", model="fake")
    assert not res.ok and "guardrails" in res.error and res.attempts == 1
    assert "orders" in assistant.db.tables()


def test_cannot_answer(assistant, fake_model):
    fake_model.replies = ["CANNOT_ANSWER: the data has no marketing spend"]
    res = assistant.ask("What was our ad spend?", model="fake")
    assert res.cannot_answer and "marketing" in res.answer and res.df is None


def test_rag_context_reaches_the_model_only_when_on(assistant, fake_model):
    q = "What percentage of delivered orders arrived late by state?"
    fake_model.replies = ["```sql\nSELECT 1 AS x FROM orders LIMIT 1\n```"]
    res = assistant.ask(q, model="fake", summarize=False, use_rag=True)
    ctx = fake_model.calls[-1]["system"][1]
    assert "late delivery" in ctx and "Verified example" in ctx and res.examples and res.terms

    fake_model.replies = ["```sql\nSELECT 1 AS x FROM orders LIMIT 1\n```"]
    res = assistant.ask(q, model="fake", summarize=False, use_rag=False)
    assert fake_model.calls[-1]["system"][1] == "" and not res.examples


def test_schema_is_always_in_the_static_prompt(assistant, fake_model):
    fake_model.replies = ["```sql\nSELECT 1 AS x FROM orders LIMIT 1\n```"]
    assistant.ask("anything", model="fake", summarize=False, use_rag=False)
    static = fake_model.calls[-1]["system"][0]
    for t in config.ALLOWED_TABLES:
        assert f"### {t}" in static


def test_end_to_end_with_gold_sql_scores_100(assistant, fake_model, gold_by_question, db):
    """A 'perfect' model that returns the gold SQL must score 40/40 - checks the whole loop."""
    correct = 0
    for question, sql in gold_by_question.items():
        fake_model.replies = [f"```sql\n{sql}\n```"]
        res = assistant.ask(question, model="fake", summarize=False)
        correct += bool(res.ok and results_match(db.run(sql).df, res.df))
    assert correct == len(gold_by_question) == 40


def test_unknown_model_is_a_clear_error():
    with pytest.raises(ValueError):
        llm.complete("x", [{"role": "user", "content": "hi"}], model="gpt-99")


def test_bm25_finds_relevant_example(retriever):
    ids = [h.item["id"] for h in retriever.examples_for("late delivery rate for each state")]
    assert "ex06" in ids
    terms = [h.item["term"] for h in retriever.terms_for("total revenue last year")]
    assert terms[0] == "revenue"


def test_embedding_backend_stores_vectors_in_duckdb(tmp_path, monkeypatch):
    """Swap in a tiny fake embedding model so the test needs no download."""
    import sys
    import types

    import duckdb

    vocab = ["late", "revenue", "customer", "review", "seller", "month", "state", "payment"]

    class FakeEmbedding:
        def __init__(self, name, cache_dir=None):
            self.name = name

        def _vec(self, t):
            v = np.array([t.lower().count(w) for w in vocab], dtype=float) + 1e-3
            return v / np.linalg.norm(v)

        def passage_embed(self, texts):
            return [self._vec(t) for t in texts]

        def query_embed(self, text):
            return iter([self._vec(text)])

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeEmbedding))
    idx = tmp_path / "rag.duckdb"
    r = Retriever(backend="embeddings", index_path=idx)
    assert r.backend == "embeddings"
    assert r.examples_for("late deliveries by state", k=3)[0].item["id"] == "ex06"

    con = duckdb.connect(str(idx), read_only=True)
    n, dtype = con.execute("SELECT COUNT(*), any_value(typeof(emb)) FROM examples").fetchone()
    con.close()
    assert n == len(r.examples) and dtype == f"FLOAT[{len(vocab)}]"

    # rebuilding with unchanged content reuses the stored vectors
    Retriever(backend="embeddings", index_path=idx)


def test_auto_backend_falls_back_to_bm25(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "fastembed", None)  # import fails
    r = Retriever(backend="auto", index_path=tmp_path / "x.duckdb")
    assert r.backend == "bm25"


def test_anthropic_request_shape(monkeypatch):
    """Checks the Claude request without calling the API: caching on the static block,
    temperature only for models that accept it, thinking blocks dropped from the answer."""
    from types import SimpleNamespace

    sent = {}

    class FakeMessages:
        def create(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="thinking", thinking="..."),
                         SimpleNamespace(type="text", text="```sql\nSELECT 1\n```")],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                      cache_read_input_tokens=2000, cache_creation_input_tokens=0),
            )

    monkeypatch.setattr(llm, "_anthropic_client", SimpleNamespace(messages=FakeMessages()))
    r = llm.complete(["STATIC", "CONTEXT"], [{"role": "user", "content": "q"}], model="claude-sonnet")
    assert r.text == "```sql\nSELECT 1\n```" and r.model == "claude-sonnet"
    assert sent["model"] == "claude-sonnet-5" and "temperature" not in sent
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in sent["system"][1]
    assert r.cost_usd == pytest.approx((10 * 2 + 2000 * 2 * 0.1 + 5 * 10) / 1e6)

    llm.complete("STATIC", [{"role": "user", "content": "q"}], model="claude-haiku")
    assert sent["model"] == "claude-haiku-4-5-20251001" and "temperature" not in sent
    assert len(sent["system"]) == 1


def test_ollama_request_shape(monkeypatch):
    from types import SimpleNamespace

    sent = {}

    def fake_post(url, json, timeout):
        sent.update(url=url, **json)
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"message": {"content": "SELECT 1"}, "prompt_eval_count": 50, "eval_count": 3})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    r = llm.complete(["STATIC", "CONTEXT"], [{"role": "user", "content": "q"}], model="qwen-coder")
    assert sent["url"].endswith("/api/chat") and sent["messages"][0]["content"] == "STATIC\n\nCONTEXT"
    assert r.text == "SELECT 1" and r.input_tokens == 50 and r.cost_usd is None
