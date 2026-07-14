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


_ROMAN = {"1": "i", "2": "ii", "3": "iii", "4": "iv", "5": "v"}
_ARABIC_SEM = re.compile(r"\b([1-5])-([1-2])\b")


def expand_query(q: str) -> str:
    """Bridge user notation to curriculum-table notation: '3-1' -> 'iii-i'.

    Curriculum docs label semesters in Roman year-sem form (III-I); users type
    arabic (3-1). Append the Roman form so BM25 can match either.
    """
    extra = [f"{_ROMAN[m.group(1)]}-{_ROMAN[m.group(2)]}" for m in _ARABIC_SEM.finditer(q)]
    return q + (" " + " ".join(extra) if extra else "")


class Retriever:
    def __init__(self, index_path: Path = INDEX_PATH):
        data = json.loads(Path(index_path).read_text(encoding="utf-8"))
        self.chunks: list[dict] = data["chunks"]
        self.source_dir: str = data.get("source_dir", "")
        self.doc_count: int = data.get("doc_count", 0)

        # Precompute per-chunk token stats + document frequencies. The doc name's
        # separators are split so a query like "ECD" matches the *name*
        # "BTech-ECD-V2" (whose compound token wouldn't otherwise match), while
        # body text keeps exact compound terms (3.2, b.tech) intact.
        self._doc_tokens: list[list[str]] = [
            tokenize(f"{c['doc'].replace('-', ' ').replace('_', ' ')} {c['text']}")
            for c in self.chunks
        ]
        self._N = len(self.chunks)
        self._df: dict[str, int] = {}
        total_len = 0
        for toks in self._doc_tokens:
            total_len += len(toks)
            for t in set(toks):
                self._df[t] = self._df.get(t, 0) + 1
        self._avgdl = total_len / self._N if self._N else 0.0

        # Optional semantic index (embeddings.json). Absent -> pure BM25.
        self._emb: list[Optional[list[float]]] = [None] * self._N
        self._emb_norm: list[float] = [0.0] * self._N
        self.has_embeddings = False
        epath = INDEX_PATH.with_name("embeddings.json")
        if epath.exists():
            try:
                by_id = json.loads(epath.read_text(encoding="utf-8")).get("by_id", {})
                for i, c in enumerate(self.chunks):
                    v = by_id.get(c["id"])
                    if v:
                        self._emb[i] = v
                        self._emb_norm[i] = math.sqrt(sum(x * x for x in v))
                self.has_embeddings = any(v is not None for v in self._emb)
            except (ValueError, OSError):
                pass

    def _idf(self, term: str) -> float:
        n = self._df.get(term, 0)
        # BM25 idf floored at 0 so common terms can't push scores negative.
        return max(0.0, math.log(1 + (self._N - n + 0.5) / (n + 0.5)))

    def _bm25_ranked(self, qset: list[str]) -> list[tuple[int, float]]:
        """All chunks with score > 0, sorted best-first: [(chunk_index, score)]."""
        scored: list[tuple[int, float]] = []
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
                scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _semantic_ranked(self, qvec: list[float]) -> list[tuple[int, float]]:
        """Chunks ranked by cosine similarity to the query vector."""
        qn = math.sqrt(sum(x * x for x in qvec)) or 1.0
        out: list[tuple[int, float]] = []
        for i, vec in enumerate(self._emb):
            if vec is None:
                continue
            dot = sum(a * b for a, b in zip(qvec, vec))
            out.append((i, dot / (qn * self._emb_norm[i]) if self._emb_norm[i] else 0.0))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def _hit(self, i: int, score: float, qset: list[str]) -> dict:
        c = self.chunks[i]
        return {
            "id": c["id"], "doc": c["doc"], "clause": c["clause"], "page": c["page"],
            "score": round(score, 4), "snippet": _snippet(c["text"], qset),
        }

    def search(self, query: str, k: int = 6) -> list[dict]:
        """Pure lexical BM25 search (used by the offline --search path)."""
        qset = list(dict.fromkeys(tokenize(expand_query(query))))
        if not qset:
            return []
        return [self._hit(i, s, qset) for i, s in self._bm25_ranked(qset)[:k]]

    def search_hybrid(self, query: str, qvec: Optional[list[float]],
                      k: int = 6) -> list[dict]:
        """Interleave BM25 and semantic top hits (BM25 first).

        Round-robin merge guarantees the strongest lexical hit (exact terms) and
        the strongest semantic hit (paraphrase/concept) both appear near the top,
        instead of a fused ranking where one signal dilutes the other. Falls back
        to pure BM25 when no query vector / no embeddings are present.
        """
        qset = list(dict.fromkeys(tokenize(expand_query(query))))
        bm = self._bm25_ranked(qset)
        if qvec is None or not self.has_embeddings:
            return [self._hit(i, s, qset) for i, s in bm[:k]]
        sem = self._semantic_ranked(qvec)
        bm_score = dict(bm)
        sem_score = dict(sem)
        order: list[int] = []
        seen: set[int] = set()
        bi = si = 0
        while len(order) < k and (bi < len(bm) or si < len(sem)):
            if bi < len(bm):
                i = bm[bi][0]; bi += 1
                if i not in seen:
                    seen.add(i); order.append(i)
            if len(order) >= k:
                break
            if si < len(sem):
                i = sem[si][0]; si += 1
                if i not in seen:
                    seen.add(i); order.append(i)
        return [self._hit(i, sem_score.get(i, bm_score.get(i, 0.0)), qset) for i in order]

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
