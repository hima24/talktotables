"""Question -> SQL -> safe execution -> plain-English answer."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from src import config, llm
from src.db import Database, QueryTimeout
from src.guardrails import UnsafeSQLError, extract_sql, validate
from src.retriever import Retriever

RULES = """You are a careful analytics engineer. You turn business questions about {dataset}
into ONE DuckDB SQL query.

Rules:
- Write a single SELECT query (CTEs are fine). Never modify data.
- Use only the tables and columns in the schema below.
- Follow the business definitions exactly when they are provided.
- Return only the columns the question asks for, plus a label column when grouping.
- Round money, averages and ratios with ROUND(x, 2) unless a definition says otherwise.
  Percentages are 0-100 (not 0-1) and rounded to 2 decimals.
- For monthly buckets return CAST(date_trunc('month', ts) AS DATE) AS month.
- For "top N" questions use ORDER BY ... LIMIT N.
- Reply with the SQL in a ```sql code block and nothing else.
- If the question can't be answered from this data, reply exactly: CANNOT_ANSWER: <short reason>
"""

ANSWER_PROMPT = """Question: {question}

SQL that was run:
{sql}

Result ({n} rows{trunc}):
{table}

Answer the question in one or two plain-English sentences using these results.
Mention specific numbers. Don't describe the SQL."""


@dataclass
class AskResult:
    question: str
    model: str
    sql: str | None = None
    df: pd.DataFrame | None = None
    answer: str | None = None
    error: str | None = None
    cannot_answer: bool = False
    attempts: int = 0
    truncated: bool = False
    used_rag: bool = True
    examples: list[dict] = field(default_factory=list)
    terms: list[dict] = field(default_factory=list)
    llm_calls: list[llm.LLMResponse] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.df is not None and self.error is None

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens + c.cache_read_tokens + c.cache_write_tokens for c in self.llm_calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)

    @property
    def cost_usd(self) -> float | None:
        costs = [c.cost_usd for c in self.llm_calls]
        return None if any(c is None for c in costs) else sum(costs)


class SQLAssistant:
    def __init__(self, db: Database | None = None, retriever: Retriever | None = None,
                 schema_text: str | None = None, allowed_tables: set[str] | None = None,
                 dataset: str = "the Olist e-commerce data"):
        """For the Olist demo, call with no extra arguments. For uploaded data, pass that data's
        Database, schema_text and allowed_tables (see src/workspace.py) and use_rag=False."""
        self.db = db or Database()
        self.retriever = retriever or Retriever()
        self.schema = schema_text or config.SCHEMA_PATH.read_text(encoding="utf-8")
        self.allowed_tables = allowed_tables  # None means the Olist tables in config.ALLOWED_TABLES
        self.dataset = dataset

    # -- prompt building -----------------------------------------------------
    def static_prompt(self) -> str:
        """Rules + full schema. Identical on every call, so it is cached."""
        return f"{RULES.format(dataset=self.dataset)}\n\n{self.schema}"

    def context_prompt(self, examples: list[dict], terms: list[dict]) -> str:
        parts = []
        if terms:
            parts.append("## Business definitions\n" + "\n".join(f"- {t['term']}: {t['definition']}" for t in terms))
        if examples:
            shots = "\n\n".join(f"Q: {e['question']}\n```sql\n{e['sql']}\n```" for e in examples)
            parts.append("## Verified example queries (similar questions)\n" + shots)
        return "\n\n".join(parts)

    # -- main loop -----------------------------------------------------------
    def ask(self, question: str, model: str = config.DEFAULT_MODEL, use_rag: bool = True,
            summarize: bool = True, max_retries: int = 1) -> AskResult:
        start = time.perf_counter()
        res = AskResult(question=question, model=model, used_rag=use_rag)
        try:
            self._ask(res, question, model, use_rag, summarize, max_retries)
        except Exception as e:  # never crash the app; report what went wrong
            res.error = f"{type(e).__name__}: {e}"
        res.seconds = time.perf_counter() - start
        return res

    def _ask(self, res, question, model, use_rag, summarize, max_retries):
        if use_rag:
            res.examples = [h.item for h in self.retriever.examples_for(question)]
            res.terms = [h.item for h in self.retriever.terms_for(question)]
        system = [self.static_prompt(), self.context_prompt(res.examples, res.terms)]
        messages = [{"role": "user", "content": question}]

        for attempt in range(1 + max_retries):
            res.attempts = attempt + 1
            reply = llm.complete(system, messages, model=model)
            res.llm_calls.append(reply)
            text = reply.text.strip()

            if "CANNOT_ANSWER" in text and "```" not in text:
                res.cannot_answer = True
                res.error = None
                res.answer = text.split("CANNOT_ANSWER:", 1)[-1].strip() or "This can't be answered from the data."
                return

            sql = extract_sql(text)
            res.sql = sql
            try:
                sql = validate(sql, self.allowed_tables)
            except UnsafeSQLError as e:
                # Unsafe SQL is never retried: we refuse rather than negotiate.
                res.error = f"Blocked by guardrails: {e}"
                return
            try:
                out = self.db.run(sql)
            except QueryTimeout as e:
                res.error = str(e)
                return
            except Exception as e:  # SQL error: give the model one chance to fix it
                res.error = f"SQL error: {str(e).splitlines()[0]}"
                messages = messages + [
                    {"role": "assistant", "content": f"```sql\n{sql}\n```"},
                    {"role": "user", "content": f"That query failed with this DuckDB error:\n{e}\n"
                                                "Fix it and reply with the corrected SQL only."},
                ]
                continue
            res.df, res.truncated, res.error = out.df, out.truncated, None
            break

        if res.ok and summarize:
            res.answer = self._summarize(res, model)

    def _summarize(self, res: AskResult, model: str) -> str:
        table = res.df.head(20).to_csv(index=False)
        prompt = ANSWER_PROMPT.format(
            question=res.question, sql=res.sql, n=len(res.df),
            trunc=", truncated" if res.truncated else "", table=table,
        )
        reply = llm.complete("You explain query results to business users, briefly and accurately.",
                             [{"role": "user", "content": prompt}], model=model, max_tokens=1024)
        res.llm_calls.append(reply)
        return reply.text.strip()
