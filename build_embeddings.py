#!/usr/bin/env python3
"""
build_embeddings.py — Offline embedding index for hybrid (BM25 + semantic) search.

Reads index.json and computes a vector per chunk via a local ollama embedding
model, writing embeddings.json keyed by chunk id. This is OPTIONAL: if
embeddings.json is absent, the query tool falls back to pure BM25.

Embeddings complement BM25 — they help conceptual / "what is X" queries where the
answer uses different words than the question (the fuzzy case lexical search
misses). Run after (re)building index.json.

    python3 build_embeddings.py            # uses index.json -> embeddings.json
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
BATCH = 64
HERE = Path(__file__).parent

# nomic-embed-text is trained with task prefixes; documents use search_document.
DOC_PREFIX = "search_document: "


def embed(texts: list[str]) -> list[list[float]]:
    payload = json.dumps({"model": MODEL, "input": texts}).encode()
    req = urllib.request.Request(
        f"{HOST}/api/embed", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
    if "embeddings" not in body:
        raise RuntimeError(body.get("error", "no embeddings returned"))
    return body["embeddings"]


def main() -> int:
    index = json.loads((HERE / "index.json").read_text(encoding="utf-8"))
    chunks = index["chunks"]
    print(f"embedding {len(chunks)} chunks with {MODEL} ...")

    by_id: dict[str, list[float]] = {}
    dim = 0
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        texts = [DOC_PREFIX + f"{c['doc']} {c['clause']}\n{c['text']}"[:2000] for c in batch]
        try:
            vecs = embed(texts)
        except Exception as e:  # noqa: BLE001
            print(f"  ! embed failed at {i}: {e}", file=sys.stderr)
            return 1
        for c, v in zip(batch, vecs):
            by_id[c["id"]] = [round(x, 5) for x in v]
            dim = len(v)
        print(f"  {min(i + BATCH, len(chunks))}/{len(chunks)}", end="\r", flush=True)

    out = {"model": MODEL, "dim": dim, "count": len(by_id), "by_id": by_id}
    (HERE / "embeddings.json").write_text(json.dumps(out))
    print(f"\nWrote embeddings.json  ({len(by_id)} vectors, dim {dim})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
