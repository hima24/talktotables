""""Use your own data": turn uploaded files into a private, read-only DuckDB database.

Flow:
    ws = build_workspace([("sales.csv", b"..."), ("stores.xlsx", b"...")])
    assistant = SQLAssistant(db=ws.db, retriever=..., schema_text=ws.schema_text,
                             allowed_tables=ws.tables, dataset="the user's uploaded data")

Each upload becomes one table (each Excel sheet becomes its own table). Names are
cleaned to snake_case so the model never has to quote them. The tables are written to
a temporary DuckDB file, which is then reopened with the same read-only, no-file-access
settings as the Olist database (src/db.py), so the same 3 guardrail layers apply.
"""
from __future__ import annotations

import io
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from src import config, llm
from src.db import Database

SUPPORTED = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet"}

# Words that make awkward table/column names in SQL
_RESERVED = {"order", "group", "select", "from", "where", "table", "user", "limit", "join", "by", "to",
             "desc", "asc", "case", "when", "end", "date", "time", "values", "index", "key", "all", "default"}


class UploadError(ValueError):
    """A file that can't be loaded (wrong type, too big, unreadable)."""


def clean_name(name: str, taken: set[str], fallback: str = "t") -> str:
    """'Sales Report (2024).csv' -> 'sales_report_2024'; unique within `taken`."""
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or fallback
    if base[0].isdigit():
        base = f"{fallback}_{base}"
    if base in _RESERVED:
        base = f"{base}_{fallback}"
    out, i = base, 2
    while out in taken:
        out, i = f"{base}_{i}", i + 1
    taken.add(out)
    return out


@dataclass
class TableInfo:
    name: str
    source: str
    rows: int
    columns: list[tuple[str, str]]  # (clean name, DuckDB type)
    renamed: dict[str, str] = field(default_factory=dict)  # original -> clean, only where they differ


@dataclass
class Workspace:
    db: Database
    tables: set[str]
    infos: list[TableInfo]
    schema_text: str
    folder: Path

    @property
    def total_rows(self) -> int:
        return sum(t.rows for t in self.infos)

    def preview(self, table: str, n: int = 5) -> pd.DataFrame:
        return self.db.run(f'SELECT * FROM "{table}" LIMIT {int(n)}').df

    def close(self) -> None:
        try:
            self.db.close()
        finally:
            shutil.rmtree(self.folder, ignore_errors=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _frames_from_file(filename: str, data: bytes, folder: Path, con) -> list[tuple[str, str]]:
    """Load one uploaded file into `con`. Returns [(temporary table name, label)]."""
    ext = Path(filename).suffix.lower()
    stem = Path(filename).stem
    loaded = []
    if ext in (".csv", ".tsv", ".txt"):
        path = folder / f"upload_{len(list(folder.iterdir()))}{ext}"
        path.write_bytes(data)
        tmp = f"_tmp_{len(con.execute('SHOW TABLES').fetchall())}"
        try:
            con.execute(f"CREATE TABLE {tmp} AS SELECT * FROM read_csv_auto(?, sample_size = -1)", [str(path)])
        except duckdb.Error:
            # fall back to pandas, which is more forgiving with odd quoting/encodings
            df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding_errors="replace")
            con.register("_df", df)
            con.execute(f"CREATE TABLE {tmp} AS SELECT * FROM _df")
            con.unregister("_df")
        loaded.append((tmp, stem))
    elif ext in (".xlsx", ".xls"):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        for sheet, df in sheets.items():
            if df.dropna(how="all").empty:
                continue
            df = df.dropna(how="all").dropna(axis=1, how="all")
            df.columns = [str(c) for c in df.columns]
            for c in df.columns:  # mixed-type object columns become text so DuckDB can store them
                if df[c].dtype == object:
                    df[c] = df[c].map(lambda v: None if pd.isna(v) else str(v))
            tmp = f"_tmp_{len(con.execute('SHOW TABLES').fetchall())}"
            con.register("_df", df)
            con.execute(f"CREATE TABLE {tmp} AS SELECT * FROM _df")
            con.unregister("_df")
            loaded.append((tmp, stem if len(sheets) == 1 else f"{stem}_{sheet}"))
    elif ext == ".parquet":
        path = folder / f"upload_{len(list(folder.iterdir()))}.parquet"
        path.write_bytes(data)
        tmp = f"_tmp_{len(con.execute('SHOW TABLES').fetchall())}"
        con.execute(f"CREATE TABLE {tmp} AS SELECT * FROM read_parquet(?)", [str(path)])
        loaded.append((tmp, stem))
    else:
        raise UploadError(f"{filename}: unsupported file type. Use CSV, TSV, Excel or Parquet.")
    return loaded


def build_workspace(files: list[tuple[str, bytes]], max_mb: float = config.MAX_UPLOAD_MB,
                    max_files: int = config.MAX_UPLOAD_FILES) -> Workspace:
    if not files:
        raise UploadError("No files uploaded.")
    if len(files) > max_files:
        raise UploadError(f"Up to {max_files} files at a time, please.")
    for name, data in files:
        if Path(name).suffix.lower() not in SUPPORTED:
            raise UploadError(f"{name}: unsupported file type. Use CSV, TSV, Excel or Parquet.")
        if len(data) > max_mb * 1024 * 1024:
            raise UploadError(f"{name} is {len(data) / 1024 / 1024:.1f} MB; the limit is {max_mb:.0f} MB per file.")

    folder = Path(tempfile.mkdtemp(prefix="talktotables_"))
    db_path = folder / "workspace.duckdb"
    con = duckdb.connect(str(db_path))  # writable only while we load; reopened read-only below
    infos: list[TableInfo] = []
    taken: set[str] = set()
    try:
        for filename, data in files:
            try:
                loaded = _frames_from_file(filename, data, folder, con)
            except UploadError:
                raise
            except Exception as e:
                raise UploadError(f"Couldn't read {filename}: {str(e).splitlines()[0]}") from e
            for tmp, label in loaded:
                table = clean_name(label, taken)
                con.execute(f'ALTER TABLE {tmp} RENAME TO "{table}"')
                cols_taken: set[str] = set()
                renamed = {}
                for (orig,) in con.execute(f'SELECT column_name FROM (DESCRIBE "{table}")').fetchall():
                    new = clean_name(orig, cols_taken, fallback="col")
                    if new != orig:
                        con.execute(f'ALTER TABLE "{table}" RENAME COLUMN "{orig}" TO "{new}"')
                        renamed[orig] = new
                cols = con.execute(f'SELECT column_name, column_type FROM (DESCRIBE "{table}")').fetchall()
                rows = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                infos.append(TableInfo(table, filename, rows, [(c, t) for c, t in cols], renamed))
        if not infos:
            raise UploadError("The uploaded files had no data.")
        schema_text = describe(con, infos)
        con.execute("CHECKPOINT")
    except Exception:
        con.close()
        shutil.rmtree(folder, ignore_errors=True)
        raise
    con.close()

    db = Database(db_path)  # read-only, no file access, row cap + timeout
    return Workspace(db=db, tables={t.name for t in infos}, infos=infos, schema_text=schema_text, folder=folder)


# ---------------------------------------------------------------------------
# Schema description for the prompt
# ---------------------------------------------------------------------------
def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


def describe(con, infos: list[TableInfo], max_cols: int = 60) -> str:
    """A schema.md-style description: types, ranges and example values, so the model
    knows what the data looks like without seeing all of it."""
    lines = ["# Uploaded data (DuckDB)", "",
             "Tables the user uploaded. Names were cleaned to snake_case, so no quoting is needed.", "",
             "## Tables"]
    for t in infos:
        lines += ["", f"### {t.name} ({t.rows:,} rows, from {t.source})"]
        for col, typ in t.columns[:max_cols]:
            q = f'"{col}"'
            detail = ""
            try:
                upper = typ.upper()
                if any(k in upper for k in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "REAL", "NUMERIC")):
                    mn, mx, avg = con.execute(f'SELECT MIN({q}), MAX({q}), AVG({q}) FROM "{t.name}"').fetchone()
                    if mn is not None:
                        detail = f"min {_fmt(mn)}, max {_fmt(mx)}, avg {_fmt(float(avg))}"
                elif any(k in upper for k in ("DATE", "TIME")):
                    mn, mx = con.execute(f'SELECT MIN({q}), MAX({q}) FROM "{t.name}"').fetchone()
                    if mn is not None:
                        detail = f"{mn} to {mx}"
                elif "VARCHAR" in upper:
                    n = con.execute(f'SELECT approx_count_distinct({q}) FROM "{t.name}"').fetchone()[0]
                    if n and n <= 15:
                        vals = [r[0] for r in con.execute(
                            f'SELECT {q}, COUNT(*) c FROM "{t.name}" WHERE {q} IS NOT NULL '
                            f'GROUP BY 1 ORDER BY c DESC LIMIT 8').fetchall()]
                        detail = f"{n} values: " + ", ".join(repr(v[:40]) for v in vals)
                    else:
                        ex = [r[0] for r in con.execute(
                            f'SELECT DISTINCT {q} FROM "{t.name}" WHERE {q} IS NOT NULL LIMIT 3').fetchall()]
                        detail = f"~{n:,} distinct, e.g. " + ", ".join(repr(v[:40]) for v in ex)
                nulls = con.execute(f'SELECT COUNT(*) - COUNT({q}) FROM "{t.name}"').fetchone()[0]
                if nulls:
                    detail += f"{'; ' if detail else ''}{nulls:,} NULL"
            except duckdb.Error:
                pass
            lines.append(f"- {col} {typ}" + (f" ({detail})" if detail else ""))
        if len(t.columns) > max_cols:
            lines.append(f"- ... and {len(t.columns) - max_cols} more columns")

    # columns that share a name across tables are likely join keys
    by_col: dict[str, list[str]] = {}
    for t in infos:
        for col, _ in t.columns:
            by_col.setdefault(col, []).append(t.name)
    joins = [(c, ts) for c, ts in by_col.items() if len(ts) > 1]
    if joins:
        lines += ["", "## Possible join keys (same column name in several tables; check before relying on them)"]
        for c, ts in joins:
            lines.append(f"- {c}: " + " = ".join(f"{t}.{c}" for t in ts))
    lines += ["", "## Notes",
              "- This data has no verified business definitions: state any assumption in column aliases.",
              "- When joining, count distinct keys to avoid double-counting (join fan-out)."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------
SUGGEST_PROMPT = """Here is a database schema:

{schema}

Suggest 6 varied business questions a manager could ask about this data. Each must be answerable
with one SQL query on these tables, and under 14 words. Mix: a total, a trend over time (if there
are dates), a top-N ranking, a comparison between groups, an average, and one using two tables if
they join. Reply with one question per line, no numbering, nothing else."""


def suggest_questions(schema_text: str, model: str = config.DEFAULT_MODEL, n: int = 6) -> list[str]:
    try:
        reply = llm.complete("You write short, concrete analytics questions.",
                             [{"role": "user", "content": SUGGEST_PROMPT.format(schema=schema_text)}],
                             model=model, max_tokens=1024)
    except Exception:
        return []
    out = []
    for line in reply.text.splitlines():
        q = re.sub(r"^\s*(?:[-*\d.)]+\s*)", "", line).strip().strip('"')
        if 8 <= len(q) <= 140 and q.endswith("?"):
            out.append(q)
    return out[:n]
