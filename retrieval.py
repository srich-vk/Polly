"""
retrieval.py — In-process BM25 over the clause-chunked policy index.

Pure stdlib (no rank_bm25 / numpy) so the whole tool runs on a bare `python3`.
Ported from the validated TypeScript engine (retrieval.ts). Rebuild index.json
with build_index.py whenever the corpus changes.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional

INDEX_PATH = Path(__file__).with_name("index.json")

# Keep clause-style tokens intact ("3.2", "b.tech", "2k21", "wi-fi") — exact
# terms are the whole point of this corpus.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")

_K1 = 1.5
_B = 0.75


def tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower())


class Retriever:
    def __init__(self, index_path: Path = INDEX_PATH):
        data = json.loads(Path(index_path).read_text(encoding="utf-8"))
        self.chunks: list[dict] = data["chunks"]
        self.source_dir: str = data.get("source_dir", "")
        self.doc_count: int = data.get("doc_count", 0)

        # Precompute per-chunk token stats + document frequencies.
        self._doc_tokens: list[list[str]] = [
            tokenize(f"{c['doc']} {c['text']}") for c in self.chunks
        ]
        self._N = len(self.chunks)
        self._df: dict[str, int] = {}
        total_len = 0
        for toks in self._doc_tokens:
            total_len += len(toks)
            for t in set(toks):
                self._df[t] = self._df.get(t, 0) + 1
        self._avgdl = total_len / self._N if self._N else 0.0

    def _idf(self, term: str) -> float:
        n = self._df.get(term, 0)
        # BM25 idf floored at 0 so common terms can't push scores negative.
        return max(0.0, math.log(1 + (self._N - n + 0.5) / (n + 0.5)))

    def search(self, query: str, k: int = 6) -> list[dict]:
        qset = list(dict.fromkeys(tokenize(query)))
        if not qset:
            return []
        scored: list[tuple[float, int]] = []
        for i, toks in enumerate(self._doc_tokens):
            length = len(toks)
            if not length:
                continue
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for term in qset:
                f = tf.get(term)
                if not f:
                    continue
                denom = f + _K1 * (1 - _B + _B * length / self._avgdl)
                score += self._idf(term) * (f * (_K1 + 1) / denom)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        hits = []
        for score, i in scored[:k]:
            c = self.chunks[i]
            hits.append({
                "id": c["id"],
                "doc": c["doc"],
                "clause": c["clause"],
                "page": c["page"],
                "score": round(score, 3),
                "snippet": _snippet(c["text"], qset),
            })
        return hits

    def read_section(self, doc: str, clause: Optional[str] = None) -> list[dict]:
        dq = doc.lower()
        hits = [c for c in self.chunks if dq in c["doc"].lower()]
        if clause and clause.strip():
            cq = clause.strip().lower()
            exact = [c for c in hits if c["clause"].lower() == cq]
            hits = exact if exact else [c for c in hits if c["clause"].lower().startswith(cq)]
        return hits[:12]

    def list_docs(self) -> list[str]:
        return sorted({c["doc"] for c in self.chunks})

    def citation_exists(self, doc: str, clause: str) -> bool:
        """True if a chunk matches this doc (substring) + clause (exact/prefix)."""
        return bool(self.read_section(doc, clause))


def _snippet(text: str, qset: list[str], width: int = 260) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    lower = flat.lower()
    at = -1
    for t in qset:
        p = lower.find(t)
        if p >= 0 and (at < 0 or p < at):
            at = p
    if at < 0:
        return flat[:width] + (" …" if len(flat) > width else "")
    start = max(0, at - width // 3)
    end = min(len(flat), start + width)
    return ("… " if start > 0 else "") + flat[start:end] + (" …" if end < len(flat) else "")


# Module-level singleton so the index is loaded once per process.
_default: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _default
    if _default is None:
        _default = Retriever()
    return _default
