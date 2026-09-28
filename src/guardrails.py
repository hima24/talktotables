"""Check that generated SQL is a single, read-only query on known tables.

This is layer 1 of 3. Layer 2 is the read-only DuckDB connection with file
access disabled (src/db.py). Layer 3 is the row limit and query timeout.
Even if this check missed something, layer 2 physically can't write or read files.
"""
from __future__ import annotations

import logging
import re

import sqlglot
from sqlglot import exp

from src import config

logging.getLogger("sqlglot").setLevel(logging.ERROR)


class UnsafeSQLError(ValueError):
    """Raised when generated SQL fails a safety check."""


# Statement types that must never run (matched by class name so this keeps
# working across sqlglot versions).
_FORBIDDEN_NODES = {
    "Insert", "Update", "Delete", "Merge", "Drop", "Create", "Alter", "AlterTable",
    "TruncateTable", "Command", "Copy", "Attach", "Detach", "Pragma", "Set", "Use",
    "Transaction", "Commit", "Rollback", "Export", "Install", "Load", "Grant", "Revoke",
}

# DuckDB table functions / functions that reach outside the database or run SQL strings.
_FORBIDDEN_FUNCTIONS = re.compile(
    r"^(read_\w+|.*_scan|glob|sniff_csv|query|query_table|getenv|pragma_\w+|duckdb_\w+|"
    r"load_extension|install_extension|current_setting|which_secret|parquet_\w+|iceberg_\w+|delta_\w+)$",
    re.IGNORECASE,
)

_ALLOWED_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply (```sql fences or bare text)."""
    m = re.search(r"```(?:sql|duckdb)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    sql = m.group(1) if m else text
    return sql.strip().rstrip(";").strip()


def validate(sql: str, allowed_tables: set[str] | None = None) -> str:
    """Return the cleaned SQL if it is safe, else raise UnsafeSQLError."""
    allowed = {t.lower() for t in (allowed_tables or config.ALLOWED_TABLES)}
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise UnsafeSQLError("Empty query.")

    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQLError(f"Could not parse SQL: {str(e).splitlines()[0]}") from e

    if len(statements) != 1:
        raise UnsafeSQLError("Only one statement is allowed.")
    tree = statements[0]

    if not isinstance(tree, _ALLOWED_ROOTS):
        raise UnsafeSQLError(f"Only SELECT queries are allowed (got {type(tree).__name__}).")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}

    for node in tree.walk():
        name = type(node).__name__
        if name in _FORBIDDEN_NODES:
            raise UnsafeSQLError(f"{name} statements are not allowed.")

        if isinstance(node, exp.Func):
            fname = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
            if fname and _FORBIDDEN_FUNCTIONS.match(fname):
                raise UnsafeSQLError(f"Function {fname}() is not allowed.")

        if isinstance(node, exp.Table):
            if not isinstance(node.this, exp.Identifier):
                # FROM read_csv(...), FROM range(...), FROM 'file.csv' etc.
                raise UnsafeSQLError("Only the Olist tables can be queried (no table functions or files).")
            db = (node.db or "").lower()
            if node.catalog or db not in ("", "main"):
                raise UnsafeSQLError("Schema- or database-qualified tables are not allowed.")
            tname = node.name.lower()
            if tname not in allowed and tname not in cte_names:
                raise UnsafeSQLError(f"Unknown table: {node.name}.")

    return sql
