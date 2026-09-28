"""Retrieval for the SQL assistant (the "R" in RAG).

Two small knowledge bases live in rag/:
  - examples.json: verified question -> SQL pairs (few-shot examples)
  - glossary.json: business definitions ("revenue", "late delivery", ...)

For each new question we fetch the most similar examples and definitions and
hand them to the model. The schema itself is NOT retrieved: 9 tables fit in
the prompt comfortably, so it is always included in full.

Backends:
  embeddings - local sentence embeddings (fastembed, BAAI/bge-small-en-v1.5),
               vectors stored in a DuckDB file and searched with cosine similarity
  bm25       - keyword scoring, no downloads needed
  auto       - embeddings if the model can be loaded, otherwise bm25
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import duckdb

from src import config


@dataclass
class Hit:
    item: dict
    score: float


def load_examples(path: Path = config.RAG_DIR / "examples.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_glossary(path: Path = config.RAG_DIR / "glossary.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def example_text(e: dict) -> str:
    return e["question"]


def term_text(t: dict) -> str:
    return f"{t['term']}. {t['keywords']}. {t['definition']}"


# ---------------------------------------------------------------------------
# BM25 (keyword) backend
# ---------------------------------------------------------------------------
_STOP = set("a an the of to in on for by and or is are was were what which who how many much show list "
            "give me each per with from that this do does did be at as it its than all any".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in toks:
        if t in _STOP:
            continue
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


# ---------------------------------------------------------------------------
# Embedding backend (vectors stored in DuckDB)
# ---------------------------------------------------------------------------
class EmbeddingIndex:
    """Embeds the knowledge base once and keeps the vectors in a DuckDB table."""

    def __init__(self, model_name: str, index_path: Path):
        from fastembed import TextEmbedding  # imported lazily: optional dependency

        self.model = TextEmbedding(model_name, cache_dir=str(config.EMBED_CACHE))
        self.model_name = model_name
        self.index_path = Path(index_path)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.passage_embed(texts)]

    def _embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()

    def build(self, name: str, texts: list[str]) -> None:
        """(Re)build table `name` only if the source texts changed."""
        digest = hashlib.sha256(("\n".join(texts) + self.model_name).encode()).hexdigest()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(self.index_path))
        try:
            con.execute("CREATE TABLE IF NOT EXISTS meta (name VARCHAR PRIMARY KEY, digest VARCHAR)")
            row = con.execute("SELECT digest FROM meta WHERE name = ?", [name]).fetchone()
            if row and row[0] == digest:
                return
            vecs = self._embed(texts)
            dim = len(vecs[0])
            con.execute(f"CREATE OR REPLACE TABLE {name} (idx INTEGER, emb FLOAT[{dim}])")
            con.executemany(f"INSERT INTO {name} VALUES (?, ?)", list(enumerate(vecs)))
            con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", [name, digest])
        finally:
            con.close()

    def search(self, name: str, query: str, k: int) -> list[tuple[int, float]]:
        q = self._embed_query(query)
        con = duckdb.connect(str(self.index_path), read_only=True)
        try:
            dim = len(q)
            rows = con.execute(
                f"SELECT idx, array_cosine_similarity(emb, ?::FLOAT[{dim}]) AS sim "
                f"FROM {name} ORDER BY sim DESC LIMIT ?",
                [q, k],
            ).fetchall()
        finally:
            con.close()
        return [(int(i), float(s)) for i, s in rows]


# ---------------------------------------------------------------------------
# Public retriever
# ---------------------------------------------------------------------------
class Retriever:
    def __init__(self, backend: str = config.RAG_BACKEND, examples: list[dict] | None = None,
                 glossary: list[dict] | None = None, index_path: Path | None = None):
        self.examples = examples if examples is not None else load_examples()
        self.glossary = glossary if glossary is not None else load_glossary()
        self.index_path = Path(index_path or config.ROOT / "data" / "rag_index.duckdb")
        self.backend = "bm25"
        self._emb = None

        if backend in ("auto", "embeddings"):
            try:
                self._emb = EmbeddingIndex(config.EMBED_MODEL, self.index_path)
                self._emb.build("examples", [example_text(e) for e in self.examples])
                self._emb.build("glossary", [term_text(t) for t in self.glossary])
                self.backend = "embeddings"
            except Exception as e:  # no fastembed installed, or model download blocked
                if backend == "embeddings":
                    raise
                print(f"[retriever] embeddings unavailable ({type(e).__name__}); using BM25")
                self._emb = None

        if self.backend == "bm25":
            self._bm25_ex = BM25([example_text(e) for e in self.examples])
            self._bm25_gl = BM25([term_text(t) for t in self.glossary])

    def _search(self, name: str, items: list[dict], query: str, k: int) -> list[Hit]:
        if k <= 0 or not items:
            return []
        if self.backend == "embeddings":
            return [Hit(items[i], s) for i, s in self._emb.search(name, query, k)]
        bm = self._bm25_ex if name == "examples" else self._bm25_gl
        scored = sorted(enumerate(bm.scores(query)), key=lambda x: -x[1])
        return [Hit(items[i], s) for i, s in scored[:k] if s > 0]

    def examples_for(self, question: str, k: int = config.TOP_K_EXAMPLES) -> list[Hit]:
        return self._search("examples", self.examples, question, k)

    def terms_for(self, question: str, k: int = config.TOP_K_TERMS) -> list[Hit]:
        return self._search("glossary", self.glossary, question, k)
