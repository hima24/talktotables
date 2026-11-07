"""Agent mode tests. Like the other tests, these use a scripted fake model: no API key, no network."""
import pytest

from src.agent import SQLAgent, tracing_enabled
from src.assistant import SQLAssistant


@pytest.fixture
def agent(db, retriever):
    return SQLAgent(db=db, retriever=retriever)


def test_review_ok_path(agent, fake_model):
    fake_model.replies = ["```sql\nSELECT COUNT(*) AS n FROM orders\n```", "OK", "There are 600 orders."]
    res = agent.ask("How many orders are there?", model="fake")
    assert res.ok and res.attempts == 1 and res.reviews == 1
    assert int(res.df.iloc[0, 0]) == 600 and res.answer == "There are 600 orders."
    assert res.steps == ["retrieve", "write_sql", "guardrails", "run_sql", "review:ok", "summarize"]


def test_review_triggers_one_revision(agent, fake_model):
    fake_model.replies = [
        "```sql\nSELECT COUNT(*) AS n FROM order_items\n```",           # runs, but counts item rows
        "REVISE: counts order items, not orders; count distinct orders instead.",
        "```sql\nSELECT COUNT(*) AS n FROM orders\n```",
    ]
    res = agent.ask("How many orders are there?", model="fake", summarize=False)
    assert res.ok and res.attempts == 2 and res.reviews == 1
    assert "distinct orders" in res.review_note
    assert "FROM orders" in res.sql
    # the reviewer's note was sent back to the model
    assert "reviewer found a problem" in fake_model.calls[2]["messages"][-1]["content"]
    assert "review:revise" in res.steps


def test_revision_then_summary(agent, fake_model):
    fake_model.replies = [
        "```sql\nSELECT COUNT(*) AS n FROM order_items\n```",
        "REVISE: count orders, not items",
        "```sql\nSELECT COUNT(*) AS n FROM orders\n```",
        "There are 600 orders.",
    ]
    res = agent.ask("How many orders are there?", model="fake")
    assert res.ok and res.answer == "There are 600 orders." and res.steps[-1] == "summarize"


def test_reviews_are_capped(agent, fake_model):
    # one review allowed: the revised query is accepted without a second review
    fake_model.replies = [
        "```sql\nSELECT COUNT(*) AS n FROM order_items\n```",
        "REVISE: wrong table",
        "```sql\nSELECT COUNT(*) AS n FROM orders\n```",
    ]
    res = agent.ask("How many orders?", model="fake", summarize=False)
    assert res.ok and res.reviews == 1 and len(fake_model.calls) == 3


def test_sql_error_is_fixed_then_reviewed(agent, fake_model):
    fake_model.replies = [
        "```sql\nSELECT COUNT(*) FROM orders WHERE no_such_column = 1\n```",
        "```sql\nSELECT COUNT(*) AS n FROM orders\n```",
        "OK",
    ]
    res = agent.ask("How many orders?", model="fake", summarize=False)
    assert res.ok and res.attempts == 2
    assert res.steps[:6] == ["retrieve", "write_sql", "guardrails", "run_sql", "write_sql", "guardrails"]
    assert "failed" in fake_model.calls[1]["messages"][-1]["content"]


def test_gives_up_after_max_fixes(agent, fake_model):
    fake_model.replies = ["```sql\nSELECT nope FROM orders\n```"] * 3
    res = agent.ask("How many orders?", model="fake", summarize=False, max_retries=2)
    assert not res.ok and res.attempts == 3 and "SQL error" in res.error


def test_failed_revision_falls_back_to_last_good_query(agent, fake_model):
    fake_model.replies = [
        "```sql\nSELECT COUNT(*) AS n FROM orders\n```",
        "REVISE: please double check",
        "```sql\nSELECT nope FROM orders\n```",
        "```sql\nSELECT still_nope FROM orders\n```",
        "```sql\nSELECT nope_again FROM orders\n```",
    ]
    res = agent.ask("How many orders?", model="fake", summarize=False)
    assert res.ok and int(res.df.iloc[0, 0]) == 600
    assert res.steps[-1] == "fallback:last_good"


def test_unsafe_sql_is_blocked_and_not_retried(agent, fake_model):
    fake_model.replies = ["```sql\nDROP TABLE orders\n```", "```sql\nSELECT 1 FROM orders\n```"]
    res = agent.ask("Delete everything", model="fake")
    assert not res.ok and "guardrails" in res.error and res.attempts == 1
    assert "orders" in agent.db.tables()


def test_cannot_answer(agent, fake_model):
    fake_model.replies = ["CANNOT_ANSWER: the data has no marketing spend"]
    res = agent.ask("What was our ad spend?", model="fake")
    assert res.cannot_answer and "marketing" in res.answer and res.df is None


def test_rag_context_reaches_the_model(agent, fake_model):
    fake_model.replies = ["```sql\nSELECT 1 AS x FROM orders LIMIT 1\n```", "OK"]
    res = agent.ask("What percentage of delivered orders arrived late by state?", model="fake", summarize=False)
    assert "late delivery" in fake_model.calls[0]["system"][1] and res.examples


def test_from_assistant_shares_the_same_database(db, retriever):
    a = SQLAssistant(db=db, retriever=retriever, dataset="test data")
    ag = SQLAgent.from_assistant(a)
    assert ag.db is db and ag.retriever is retriever and ag.dataset == "test data"


def test_tracing_is_off_without_keys(monkeypatch):
    for k in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert not tracing_enabled()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    assert tracing_enabled()


def test_gold_sql_still_scores_40_of_40(agent, fake_model, gold_by_question, db):
    """A perfect model plus a reviewer that always says OK must score 40/40 in agent mode too."""
    from eval.compare import results_match
    correct = 0
    for question, sql in gold_by_question.items():
        fake_model.replies = [f"```sql\n{sql}\n```", "OK"]
        res = agent.ask(question, model="fake", summarize=False)
        correct += bool(res.ok and results_match(db.run(sql).df, res.df))
    assert correct == 40
