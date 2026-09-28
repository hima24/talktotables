"""Decide whether a predicted result matches the gold result ("execution accuracy").

We compare RESULTS, not SQL text: two different queries that return the same
answer both count as correct.

Rules (kept deliberately simple so they are easy to explain):
- Same number of rows.
- Every gold column must appear somewhere in the prediction. Extra columns in the
  prediction are fine (e.g. returning a category AND its revenue when only the
  category was asked for). Column names don't matter.
- Row order doesn't matter.
- Numbers match within 0.011 (so rounding to 2 decimals is fine).
- Dates and midnight timestamps compare as YYYY-MM-DD; "2017-03" is read as 2017-03-01.
"""
from __future__ import annotations

import datetime as dt
import itertools
import math
import re

import pandas as pd

TOL = 0.011
_MONTH = re.compile(r"^\d{4}-\d{2}$")


def norm(v):
    """Turn one cell into a comparable Python value."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if v is pd.NaT:
        return None
    if isinstance(v, (pd.Timestamp, dt.datetime)):
        if v.hour == 0 and v.minute == 0 and v.second == 0:
            return v.strftime("%Y-%m-%d")
        return v.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, dt.date):
        return v.isoformat()
    if isinstance(v, bool):
        return int(v)
    if hasattr(v, "item"):  # numpy scalars
        v = v.item()
    if isinstance(v, (int, float)):
        return float(v)
    if hasattr(v, "is_integer"):  # Decimal
        return float(v)
    s = str(v).strip()
    if _MONTH.match(s):
        return s + "-01"
    return s


def _key(v):
    if v is None:
        return (0, "")
    if isinstance(v, float):
        return (1, v)
    return (2, str(v))


def _eq(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= TOL + 1e-9 * max(abs(a), abs(b))
    return a == b


def _col(df: pd.DataFrame, i: int) -> list:
    return [norm(v) for v in df.iloc[:, i].tolist()]


def _same_multiset(a: list, b: list) -> bool:
    return all(_eq(x, y) for x, y in zip(sorted(a, key=_key), sorted(b, key=_key)))


def results_match(gold: pd.DataFrame, pred: pd.DataFrame, max_combos: int = 5000) -> bool:
    if pred is None:
        return False
    if len(gold) != len(pred):
        return False
    if gold.shape[1] == 0:
        return True
    if len(gold) == 0:
        return True  # both empty

    gold_cols = [_col(gold, i) for i in range(gold.shape[1])]
    pred_cols = [_col(pred, j) for j in range(pred.shape[1])]

    # Which prediction columns could stand in for each gold column?
    candidates = []
    for g in gold_cols:
        c = [j for j, p in enumerate(pred_cols) if _same_multiset(g, p)]
        if not c:
            return False
        candidates.append(c)

    gold_rows = sorted(zip(*gold_cols), key=lambda r: tuple(_key(v) for v in r))
    for n, combo in enumerate(itertools.product(*candidates)):
        if n >= max_combos:
            break
        if len(set(combo)) != len(combo):
            continue
        pred_rows = sorted(zip(*(pred_cols[j] for j in combo)), key=lambda r: tuple(_key(v) for v in r))
        if all(all(_eq(a, b) for a, b in zip(gr, pr)) for gr, pr in zip(gold_rows, pred_rows)):
            return True
    return False
