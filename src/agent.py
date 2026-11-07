"""Agent mode: the same question -> SQL -> answer pipeline, rebuilt as a LangGraph state machine.

Why an agent? The plain assistant (src/assistant.py) makes one straight pass and
can only fix SQL that *crashes*. The worst mistakes in the benchmark were queries
that ran fine but returned the wrong numbers (for example the join fan-out in
question h02). Agent mode adds a review step: after the SQL runs, the model
checks its own query and result against the question and can send it back for
one revision.

The graph:

    retrieve -> write_sql -> guardrails -> run_sql -> review -> summarize -> END
                   ^                          |          |
                   |---- fix SQL error -------|          |
                   |---- revise after review ------------|

Each box is a node (a plain Python function that reads and updates the shared
state). The arrows that branch are "conditional edges": small functions that look
at the state and pick the next node. Safety rules are unchanged: unsafe SQL is
blocked and never retried, and every query still runs on the read-only database.

Tracing: if LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set, every run is
sent to LangSmith with one span per node and per model call (prompt, reply,
tokens, latency). Without those variables, tracing is off and nothing leaves
the machine.
"""
from __future__ import annotations

import os
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src import config, llm
from src.assistant import ANSWER_PROMPT, AskResult, SQLAssistant
from src.db import QueryTimeout
from src.guardrails import UnsafeSQLError, extract_sql, validate

try:  # LangSmith is optional: without it, traceable() is a do-nothing decorator.
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*_args, **_kwargs):
        def wrap(fn):
            return fn
        return wrap


REVIEW_PROMPT = """You are reviewing a DuckDB query written to answer a business question.

Question: {question}

SQL:
{sql}

Result ({n} rows{trunc}, first rows shown):
{table}

Check for these common mistakes:
1. Join fan-out: joining one-to-many tables (e.g. orders to order_items) and then
   counting or summing per row, so values are double counted.
2. Wrong filter: a missing or extra WHERE condition (e.g. order status, date range).
3. Wrong grain or grouping compared with what was asked.
4. Percentages not on a 0-100 scale, or ratios with the wrong denominator.
5. Columns that don't answer the question, or a result that is empty or implausible.

If the query correctly answers the question, reply exactly: OK
Otherwise reply: REVISE: <one sentence saying what is wrong and how to fix it>"""


class AgentState(TypedDict, total=False):
    question: str
    model: str
    use_rag: bool
    summarize: bool
    max_fixes: int
    max_reviews: int
    examples: list[dict]
    terms: list[dict]
    system: list[str]
    messages: list[dict]
    sql: str | None
    df: Any
    truncated: bool
    error: str | None
    cannot_answer: bool
    answer: str | None
    attempts: int       # how many times the model wrote SQL
    fixes: int          # how many SQL errors were sent back
    reviews: int        # how many review steps ran
    review_note: str | None
    route: str          # set by a node, read by the conditional edge after it
    best: dict | None   # last query that ran successfully, kept as a fallback
    llm_calls: list
    steps: list[str]


@traceable(run_type="llm", name="llm.complete")
def _call_model(system, messages, model: str, max_tokens: int = 4096) -> llm.LLMResponse:
    """One traced model call. LangSmith records the prompt, reply and token counts."""
    return llm.complete(system, messages, model=model, max_tokens=max_tokens)


class SQLAgent(SQLAssistant):
    """Drop-in replacement for SQLAssistant with self-correction and self-review.

    ask() takes the same arguments and returns the same AskResult, so the app and
    the eval runner can switch between the two with one flag.
    """

    def __init__(self, *args, max_reviews: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_reviews = max_reviews
        self.graph = self._build_graph()

    @classmethod
    def from_assistant(cls, a: SQLAssistant, **kwargs) -> "SQLAgent":
        """Wrap an existing assistant (same database, retriever and schema)."""
        return cls(db=a.db, retriever=a.retriever, schema_text=a.schema,
                   allowed_tables=a.allowed_tables, dataset=a.dataset, **kwargs)

    # -- graph ---------------------------------------------------------------
    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("write_sql", self._write_sql)
        g.add_node("guardrails", self._guardrails)
        g.add_node("run_sql", self._run_sql)
        g.add_node("review", self._review)
        g.add_node("summarize", self._summarize_node)

        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "write_sql")
        g.add_conditional_edges("write_sql", route, {"guardrails": "guardrails", "end": END})
        g.add_conditional_edges("guardrails", route, {"run_sql": "run_sql", "end": END})
        g.add_conditional_edges("run_sql", route, {"write_sql": "write_sql", "review": "review",
                                                     "summarize": "summarize", "end": END})
        g.add_conditional_edges("review", route, {"write_sql": "write_sql", "summarize": "summarize", "end": END})
        g.add_edge("summarize", END)
        return g.compile()

    # -- nodes ---------------------------------------------------------------
    # Each node returns only the keys it changes; LangGraph merges them into the state.
    def _retrieve(self, s: AgentState) -> dict:
        examples, terms = [], []
        if s["use_rag"]:
            examples = [h.item for h in self.retriever.examples_for(s["question"])]
            terms = [h.item for h in self.retriever.terms_for(s["question"])]
        return {
            "examples": examples, "terms": terms,
            "system": [self.static_prompt(), self.context_prompt(examples, terms)],
            "messages": [{"role": "user", "content": s["question"]}],
            "steps": s["steps"] + ["retrieve"],
        }

    def _write_sql(self, s: AgentState) -> dict:
        reply = _call_model(s["system"], s["messages"], s["model"])
        text = reply.text.strip()
        out = {"attempts": s["attempts"] + 1, "llm_calls": s["llm_calls"] + [reply],
               "steps": s["steps"] + ["write_sql"]}
        if "CANNOT_ANSWER" in text and "```" not in text:
            out.update(cannot_answer=True, error=None, route="end",
                       answer=text.split("CANNOT_ANSWER:", 1)[-1].strip() or "This can't be answered from the data.")
            return out
        out.update(sql=extract_sql(text), route="guardrails")
        return out

    def _guardrails(self, s: AgentState) -> dict:
        try:
            sql = validate(s["sql"], self.allowed_tables)
        except UnsafeSQLError as e:
            # Unsafe SQL ends the run: we refuse rather than negotiate.
            return {"error": f"Blocked by guardrails: {e}", "df": None, "best": None,
                    "route": "end", "steps": s["steps"] + ["guardrails:blocked"]}
        return {"sql": sql, "route": "run_sql", "steps": s["steps"] + ["guardrails"]}

    def _run_sql(self, s: AgentState) -> dict:
        steps = s["steps"] + ["run_sql"]
        try:
            out = self.db.run(s["sql"])
        except QueryTimeout as e:
            return {"error": str(e), "route": "end", "steps": steps}
        except Exception as e:  # SQL error: send the error back to the model
            err = f"SQL error: {str(e).splitlines()[0]}"
            if s["fixes"] >= s["max_fixes"]:
                return {"error": err, "route": "end", "steps": steps}
            feedback = [
                {"role": "assistant", "content": f"```sql\n{s['sql']}\n```"},
                {"role": "user", "content": f"That query failed with this DuckDB error:\n{e}\n"
                                            "Fix it and reply with the corrected SQL only."},
            ]
            return {"error": err, "fixes": s["fixes"] + 1, "messages": s["messages"] + feedback,
                    "route": "write_sql", "steps": steps}
        best = {"sql": s["sql"], "df": out.df, "truncated": out.truncated}
        return {"df": out.df, "truncated": out.truncated, "error": None, "best": best,
                "route": "review" if s["reviews"] < s["max_reviews"] else ("summarize" if s["summarize"] else "end"),
                "steps": steps}

    def _review(self, s: AgentState) -> dict:
        df = s["df"]
        prompt = REVIEW_PROMPT.format(question=s["question"], sql=s["sql"], n=len(df),
                                      trunc=", truncated" if s["truncated"] else "",
                                      table=df.head(10).to_csv(index=False) or "(no rows)")
        reply = _call_model("You check SQL for correctness. Be strict but don't invent problems.",
                            [{"role": "user", "content": prompt}], s["model"], max_tokens=512)
        text = reply.text.strip()
        out = {"reviews": s["reviews"] + 1, "llm_calls": s["llm_calls"] + [reply]}
        if text.upper().startswith("REVISE"):
            note = text.split(":", 1)[-1].strip() or text
            feedback = [
                {"role": "assistant", "content": f"```sql\n{s['sql']}\n```"},
                {"role": "user", "content": f"A reviewer found a problem with that query: {note}\n"
                                            "Reply with the corrected SQL only."},
            ]
            out.update(review_note=note, messages=s["messages"] + feedback, route="write_sql",
                       steps=s["steps"] + ["review:revise"])
            return out
        out.update(route="summarize" if s["summarize"] else "end", steps=s["steps"] + ["review:ok"])
        return out

    def _summarize_node(self, s: AgentState) -> dict:
        prompt = ANSWER_PROMPT.format(question=s["question"], sql=s["sql"], n=len(s["df"]),
                                      trunc=", truncated" if s["truncated"] else "",
                                      table=s["df"].head(20).to_csv(index=False))
        reply = _call_model("You explain query results to business users, briefly and accurately.",
                            [{"role": "user", "content": prompt}], s["model"], max_tokens=1024)
        return {"answer": reply.text.strip(), "llm_calls": s["llm_calls"] + [reply],
                "steps": s["steps"] + ["summarize"]}

    # -- public API ----------------------------------------------------------
    def ask(self, question: str, model: str = config.DEFAULT_MODEL, use_rag: bool = True,
            summarize: bool = True, max_retries: int = 2) -> AskResult:
        start = time.perf_counter()
        res = AskResult(question=question, model=model, used_rag=use_rag)
        state: AgentState = {
            "question": question, "model": model, "use_rag": use_rag, "summarize": summarize,
            "max_fixes": max_retries, "max_reviews": self.max_reviews,
            "sql": None, "df": None, "truncated": False, "error": None, "cannot_answer": False,
            "answer": None, "attempts": 0, "fixes": 0, "reviews": 0, "review_note": None,
            "best": None, "llm_calls": [], "steps": [],
        }
        run_config = {
            "run_name": "talktotables-agent",
            "tags": [model, "rag-on" if use_rag else "rag-off"],
            "metadata": {"model": model, "use_rag": use_rag, "dataset": self.dataset},
            "recursion_limit": 25,
        }
        try:
            final = self.graph.invoke(state, config=run_config)
            self._fill(res, final, summarize, model)
        except Exception as e:  # never crash the app; report what went wrong
            res.error = f"{type(e).__name__}: {e}"
        res.seconds = time.perf_counter() - start
        return res

    def _fill(self, res: AskResult, s: dict, summarize: bool, model: str) -> None:
        res.sql, res.df, res.truncated = s.get("sql"), s.get("df"), s.get("truncated", False)
        res.error, res.cannot_answer, res.answer = s.get("error"), s.get("cannot_answer", False), s.get("answer")
        res.attempts, res.examples, res.terms = s.get("attempts", 0), s.get("examples", []), s.get("terms", [])
        res.llm_calls, res.steps = s.get("llm_calls", []), s.get("steps", [])
        res.reviews, res.review_note = s.get("reviews", 0), s.get("review_note")
        # A revision that then failed shouldn't throw away a query that already worked.
        best = s.get("best")
        if res.error and best and not res.error.startswith("Blocked"):
            res.sql, res.df, res.truncated, res.error = best["sql"], best["df"], best["truncated"], None
            res.steps = res.steps + ["fallback:last_good"]
            if summarize and not res.answer:
                res.answer = self._summarize(res, model)


def route(s: AgentState) -> str:
    """Conditional edge: each node sets state['route'] to the name of the next step."""
    return s["route"]


def tracing_enabled() -> bool:
    """True when runs are being sent to LangSmith."""
    on = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "")).lower() == "true"
    return on and bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
