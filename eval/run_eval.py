"""Benchmark the assistant: models x (with / without retrieval) on 40 questions.

Examples (PowerShell, from the project folder):
    python -m eval.run_eval --check-gold                  # look at every gold answer first
    python -m eval.run_eval --models claude-haiku --limit 5   # cheap smoke test
    python -m eval.run_eval --models claude-haiku claude-sonnet --rag both
    python -m eval.run_eval --models gpt-oss-120b --rag on  # open-weight model via Groq (GROQ_API_KEY)
    python -m eval.run_eval --models qwen-coder --rag on  # open-source model via Ollama

Writes eval/results/run_<time>.json (every question) and eval/results/summary.json
+ summary.md (the tables the app and README use).

Runs ADD UP: summary.json keeps the latest result for every model + retrieval setup, so
benchmarking a new model doesn't hide earlier ones. A new run replaces only the rows for the
setups it ran. Use --fresh to start the summary over.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src import config
from src.assistant import SQLAssistant
from src.db import Database
from src.retriever import Retriever, tokenize, load_examples
from eval.compare import results_match

QUESTIONS_PATH = config.EVAL_DIR / "questions.json"
RESULTS_DIR = config.EVAL_DIR / "results"
DIFFICULTIES = ["easy", "medium", "hard"]


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def leakage_report(questions: list[dict], examples: list[dict], threshold: float = 0.7) -> list[tuple]:
    """Eval questions that look too much like a retrieval example (would inflate RAG scores)."""
    flagged = []
    for q in questions:
        qt = set(tokenize(q["question"]))
        for e in examples:
            et = set(tokenize(e["question"]))
            jac = len(qt & et) / max(len(qt | et), 1)
            if jac >= threshold:
                flagged.append((q["id"], e["id"], round(jac, 2)))
    return flagged


def check_gold(db: Database, questions: list[dict]) -> None:
    for q in questions:
        print(f"\n[{q['id']}] ({q['difficulty']}) {q['question']}")
        print(f"  {q['sql']}")
        try:
            out = db.run(q["sql"])
            print(out.df.head(8).to_string(index=False))
            if len(out.df) > 8:
                print(f"  ... {len(out.df)} rows")
        except Exception as e:
            print(f"  !! GOLD SQL FAILED: {e}")


def summarize(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in records:
        groups[(r["model"], r["rag"])].append(r)
    rows = []
    for (model, rag), rs in sorted(groups.items()):
        row = {
            "model": model,
            "label": config.MODELS[model]["label"],
            "rag": rag,
            "n": len(rs),
            "accuracy": round(100 * sum(r["correct"] for r in rs) / len(rs), 1),
            "avg_seconds": round(sum(r["seconds"] for r in rs) / len(rs), 2),
            "errors": sum(1 for r in rs if r["error"]),
            "retries_used": sum(1 for r in rs if r["attempts"] > 1),
        }
        for d in DIFFICULTIES:
            sub = [r for r in rs if r["difficulty"] == d]
            row[d] = round(100 * sum(r["correct"] for r in sub) / len(sub), 1) if sub else None
        costs = [r["cost_usd"] for r in rs]
        row["cost_per_100_questions_usd"] = None if any(c is None for c in costs) else round(100 * sum(costs) / len(rs), 3)
        rows.append(row)
    return {"generated_at": datetime.now().isoformat(timespec="seconds"), "rows": rows}


def merge_with_previous(summary: dict, run_file: str, previous: dict | None) -> dict:
    """Keep earlier setups' rows (each pointing at its own run file) unless this run re-ran them."""
    for row in summary["rows"]:
        row["run_file"] = run_file
    if previous:
        done = {(r["model"], r["rag"]) for r in summary["rows"]}
        for row in previous.get("rows", []):
            if (row["model"], row["rag"]) not in done:
                row.setdefault("run_file", previous.get("run_file"))
                summary["rows"].append(row)
    summary["rows"].sort(key=lambda r: (-r["accuracy"], r["label"], not r["rag"]))
    summary["run_file"] = run_file  # latest run (older app versions read this)
    return summary


def pct(v) -> str:
    return "-" if v is None else f"{v}%"


def to_markdown(summary: dict) -> str:
    lines = [
        "| Model | Retrieval | Overall | Easy | Medium | Hard | Avg seconds | Cost / 100 questions |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        cost = "free (local)" if r["cost_per_100_questions_usd"] is None else f"${r['cost_per_100_questions_usd']:.2f}"
        lines.append(
            f"| {r['label']} | {'on' if r['rag'] else 'off'} | **{r['accuracy']}%** ({r['n']} q) | "
            f"{pct(r['easy'])} | {pct(r['medium'])} | {pct(r['hard'])} | {r['avg_seconds']} | {cost} |"
        )
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["claude-haiku"], choices=list(config.MODELS))
    ap.add_argument("--rag", choices=["on", "off", "both"], default="both")
    ap.add_argument("--ids", nargs="*", help="only these question ids")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--db", default=config.DB_PATH)
    ap.add_argument("--check-gold", action="store_true", help="print every gold answer and exit")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="replace the whole summary instead of adding to it")
    args = ap.parse_args(argv)

    questions = load_questions()
    if args.ids:
        questions = [q for q in questions if q["id"] in set(args.ids)]
    if args.limit:
        questions = questions[: args.limit]

    db = Database(args.db)
    if args.check_gold:
        check_gold(db, questions)
        return

    leaks = leakage_report(questions, load_examples())
    if leaks:
        print(f"WARNING: eval questions very similar to retrieval examples: {leaks}")

    gold = {}
    for q in questions:
        try:
            gold[q["id"]] = db.run(q["sql"]).df
        except Exception as e:
            sys.exit(f"Gold SQL for {q['id']} failed: {e}. Fix eval/questions.json first.")

    assistant = SQLAssistant(db=db, retriever=Retriever())
    print(f"Retrieval backend: {assistant.retriever.backend}")
    rag_modes = {"on": [True], "off": [False], "both": [False, True]}[args.rag]

    records = []
    for model in args.models:
        for use_rag in rag_modes:
            hits = 0
            print(f"\n== {config.MODELS[model]['label']} | retrieval {'on' if use_rag else 'off'} ==")
            for i, q in enumerate(questions, 1):
                res = assistant.ask(q["question"], model=model, use_rag=use_rag, summarize=False)
                correct = bool(res.ok and results_match(gold[q["id"]], res.df))
                hits += correct
                records.append({
                    "id": q["id"], "difficulty": q["difficulty"], "question": q["question"],
                    "model": model, "rag": use_rag, "correct": correct,
                    "sql": res.sql, "gold_sql": q["sql"], "error": res.error,
                    "cannot_answer": res.cannot_answer, "attempts": res.attempts,
                    "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                    "cost_usd": res.cost_usd, "seconds": round(res.seconds, 2),
                    "retrieved_examples": [e["id"] for e in res.examples],
                    "retrieved_terms": [t["term"] for t in res.terms],
                    "gold_preview": gold[q["id"]].head(5).astype(str).values.tolist(),
                    "pred_preview": res.df.head(5).astype(str).values.tolist() if res.df is not None else None,
                })
                mark = "ok " if correct else ("ERR" if res.error else "x  ")
                print(f"  [{mark}] {q['id']} {q['question'][:70]}")
                time.sleep(0.2)
            print(f"  -> {hits}/{len(questions)} correct")

    summary = summarize(records)
    print("\n" + to_markdown(summary))
    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_file = f"run_{stamp}.json"
        (RESULTS_DIR / run_file).write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
        summary_path = RESULTS_DIR / "summary.json"
        previous = None
        if summary_path.exists() and not args.fresh:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = merge_with_previous(summary, run_file, previous)
        md = to_markdown(summary)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (RESULTS_DIR / "summary.md").write_text(md + "\n", encoding="utf-8")
        print(f"\nSaved eval/results/{run_file}. Summary now has {len(summary['rows'])} setup(s):\n{md}")


if __name__ == "__main__":
    main()
