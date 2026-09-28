import datetime as dt
import json

import pandas as pd
import pytest

from eval.compare import results_match
from eval.run_eval import leakage_report, load_questions, summarize
from src.guardrails import validate
from src.retriever import load_examples

ROOT_QS = load_questions()
EXAMPLES = load_examples()


def test_forty_questions_with_unique_ids():
    assert len(ROOT_QS) == 40
    assert len({q["id"] for q in ROOT_QS}) == 40
    assert {q["difficulty"] for q in ROOT_QS} == {"easy", "medium", "hard"}


@pytest.mark.parametrize("item", ROOT_QS + EXAMPLES, ids=lambda x: x["id"])
def test_every_gold_and_example_query_is_safe_and_runs(db, item):
    db.run(validate(item["sql"]))


def test_no_eval_question_is_copied_into_the_examples():
    assert leakage_report(ROOT_QS, EXAMPLES, threshold=0.7) == []


def df(rows, cols):
    return pd.DataFrame(rows, columns=cols)


def test_match_ignores_row_order_and_names_and_extra_columns():
    gold = df([["a"], ["b"]], ["category"])
    pred = df([["b", 10.0], ["a", 20.0]], ["name", "revenue"])
    assert results_match(gold, pred)


def test_match_numeric_tolerance():
    assert results_match(df([[4.087]], ["x"]), df([[4.09]], ["avg"]))
    assert not results_match(df([[4.087]], ["x"]), df([[4.2]], ["avg"]))


def test_match_dates_and_month_strings():
    gold = df([[dt.date(2017, 1, 1), 5], [dt.date(2017, 2, 1), 7]], ["month", "n"])
    pred = df([["2017-02", 7], ["2017-01", 5]], ["m", "orders"])
    assert results_match(gold, pred)
    pred_ts = df([[pd.Timestamp("2017-01-01"), 5], [pd.Timestamp("2017-02-01"), 7]], ["m", "n"])
    assert results_match(gold, pred_ts)


def test_match_requires_same_pairing_not_just_same_columns():
    gold = df([["a", 1.0], ["b", 2.0]], ["k", "v"])
    swapped = df([["a", 2.0], ["b", 1.0]], ["k", "v"])
    assert not results_match(gold, swapped)


def test_match_row_count_and_missing_columns():
    assert not results_match(df([[1], [2]], ["x"]), df([[1]], ["x"]))
    assert not results_match(df([["a", 1]], ["k", "v"]), df([["a"]], ["k"]))
    assert not results_match(df([[1]], ["x"]), None)


def test_summary_by_difficulty():
    recs = [
        {"model": "claude-haiku", "rag": True, "difficulty": "easy", "correct": True, "seconds": 1, "error": None,
         "attempts": 1, "cost_usd": 0.001},
        {"model": "claude-haiku", "rag": True, "difficulty": "hard", "correct": False, "seconds": 3, "error": None,
         "attempts": 2, "cost_usd": 0.003},
    ]
    s = summarize(recs)["rows"][0]
    assert s["accuracy"] == 50.0 and s["easy"] == 100.0 and s["hard"] == 0.0 and s["medium"] is None
    assert s["cost_per_100_questions_usd"] == 0.2
    json.dumps(s)
