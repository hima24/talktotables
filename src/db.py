"""Read-only access to the Olist DuckDB file, with a row cap and a timeout."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from src import config


class QueryTimeout(RuntimeError):
    pass


@dataclass
class QueryResult:
    df: pd.DataFrame
    truncated: bool
    seconds: float


class Database:
    """A read-only DuckDB connection.

    - read_only=True: no INSERT/UPDATE/DROP can succeed, whatever the SQL says
    - enable_external_access=False: no reading or writing files, URLs, or extensions
    - lock_configuration=True: SQL can't switch those settings back on
    """

    def __init__(self, path: Path | str = config.DB_PATH, max_rows: int = config.MAX_ROWS,
                 timeout_s: float = config.QUERY_TIMEOUT_S):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python -m src.load_data")
        self.path = path
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self.con = duckdb.connect(
            str(path),
            read_only=True,
            config={"enable_external_access": False, "lock_configuration": True, "threads": 4},
        )

    def tables(self) -> list[str]:
        return [r[0] for r in self.con.execute("SHOW TABLES").fetchall()]

    def run(self, sql: str) -> QueryResult:
        """Run a query that already passed guardrails.validate()."""
        with self._lock:  # one DuckDB connection is not safe to share across threads
            timer = threading.Timer(self.timeout_s, self.con.interrupt)
            start = time.perf_counter()
            timer.start()
            try:
                # .limit() wraps the query, so the model's ORDER BY is kept
                df = self.con.sql(sql).limit(self.max_rows + 1).df()
            except duckdb.InterruptException as e:
                raise QueryTimeout(f"Query took longer than {self.timeout_s:.0f}s and was stopped.") from e
            finally:
                timer.cancel()
            seconds = time.perf_counter() - start
        truncated = len(df) > self.max_rows
        return QueryResult(df=df.head(self.max_rows), truncated=truncated, seconds=seconds)

    def close(self):
        self.con.close()
