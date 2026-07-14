#!/usr/bin/env python3
"""
build_index.py — Offline indexer for the Local Policy Doc Query System.

Extracts every PDF in the guideline corpus, splits each doc into chunks on
STRUCTURAL boundaries (clause / section numbers), and writes a single
index.json consumed at query time by the opencode `search_docs` /
`read_section` tools.

Design notes (see README):
  * Chunking is by clause/section number, never a fixed token window — policy
    docs are numbered and mid-clause splits wreck retrieval + citation quality.
  * Citations need doc name + clause/section. Where a doc has no detectable
    numbering, we fall back to page-level chunks ("page-N") so every chunk is
    still citable.
  * Only stdlib + `pdftotext` (poppler) are used — no Python query-time dep.

Usage:
  python3 build_index.py \
      --docs "/home/srich-vk/Desktop/IIIT RAG/College Guidelines" \
      --out  index.json
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HAS_OCR = bool(shutil.which("pdftoppm") and shutil.which("tesseract"))
# A page with fewer real chars than this is treated as empty -> OCR candidate.
OCR_MIN_CHARS = 20

# A heading is a line that begins a new numbered clause/section. Kept
# deliberately conservative to avoid shredding running text into noise.
HEADING_PATTERNS = [
    # 3   3.2   3.2.1   10.4.5  followed by a capitalised word / title text
    re.compile(r"^\s*(\d+(?:\.\d+){0,4})[.)]?\s+(?=[A-Z(\"'])"),
    # "Clause 4", "Section 4.2", "Rule 7", "Article III", "Annexure B", ...
    re.compile(
        r"^\s*(?:Clause|Section|Rule|Article|Chapter|Annexure|Appendix|Schedule)\s+"
        r"([A-Za-z0-9][A-Za-z0-9.\-]*)",
        re.IGNORECASE,
    ),
]

MAX_HEADING_LEN = 120  # a "heading" line longer than this is really body text


def ocr_page(pdf_path: Path, page_no: int) -> str:
    """Rasterize one page at 300dpi and OCR it with tesseract."""
    with tempfile.TemporaryDirectory() as td:
        stem = os.path.join(td, "pg")
        subprocess.run(
            ["pdftoppm", "-q", "-r", "300", "-f", str(page_no), "-l", str(page_no),
             "-png", str(pdf_path), stem],
            timeout=180, check=False,
        )
        pngs = sorted(Path(td).glob("pg*.png"))
        if not pngs:
            return ""
        out = subprocess.run(
            ["tesseract", str(pngs[0]), "-", "--psm", "3", "-l", "eng"],
            capture_output=True, text=True, timeout=180,
        )
        return out.stdout


def extract_pages(pdf_path: Path) -> list[str]:
    """Return the doc's text as a list of pages (pdftotext, form-feed split).

    Pages that yield no text layer are OCR'd on the fly (many policy PDFs in
    this corpus are scanned images — see README). OCR is per-page so a doc that
    is only partially scanned still keeps its native text where it exists.
    """
    try:
        paged = subprocess.run(
            ["pdftotext", "-q", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=120,
        ).stdout
    except Exception as e:  # noqa: BLE001
        print(f"  ! extract failed for {pdf_path.name}: {e}", file=sys.stderr)
        return []
    pages = paged.split("\f")
    if not HAS_OCR:
        return pages
    for i, page in enumerate(pages):
        if len(page.strip()) < OCR_MIN_CHARS:
            text = ocr_page(pdf_path, i + 1)
            if len(text.strip()) >= OCR_MIN_CHARS:
                pages[i] = text
                print(f"    (OCR) {pdf_path.name} p{i + 1}: +{len(text.strip())} chars")
    return pages


def match_heading(line: str, prev_line: str = "") -> "str | None":
    """Numbered/keyword clause heading (e.g. '3.2 Attendance', 'Section 4')."""
    if len(line) > MAX_HEADING_LEN:
        return None
    for pat in HEADING_PATTERNS:
        m = pat.match(line)
        if m:
            return m.group(1)
    return None


# Small words that don't count toward a line looking "title-cased".
_STOP = {"a", "an", "the", "of", "to", "for", "and", "or", "in", "on", "at",
         "by", "with", "as", "is", "are", "be", "per", "from", "into", "under", "over"}


def match_text_heading(line: str, prev_line: str = "") -> "str | None":
    """Unnumbered *titled* section heading, e.g. 'Minimum Credits to be taken in
    a Semester' or 'Guidelines for MS & PhD students:'.

    Deliberately conservative and only used as a second pass for docs that have
    no numbered headings at all, so it can't shred well-structured docs.
    """
    s = line.strip()
    if not (3 <= len(s) <= 80):
        return None
    words = s.split()
    if not (2 <= len(words) <= 12):
        return None
    if s[-1] in ".,;":                       # sentence-like → not a heading
        return None
    if s.endswith(":"):                      # trailing colon is a strong heading signal
        return s[:-1].strip()[:60]
    # Otherwise require Title-Case / ALL-CAPS and a clean break from prior text.
    sig = [w for w in words if w[:1].isalpha() and w.lower() not in _STOP]
    if not sig:
        return None
    title_like = (sum(w[:1].isupper() for w in sig) / len(sig) >= 0.7) or s.isupper()
    prev = prev_line.strip()
    clean_break = prev == "" or prev.endswith((".", ":", "?"))
    if title_like and clean_break:
        return s[:60]
    return None


def _accumulate(doc: str, pages: list[str], detector) -> list[dict]:
    """Walk lines, starting a new chunk whenever `detector(line, prev)` fires."""
    chunks: list[dict] = []
    cur = None

    def flush():
        nonlocal cur
        if cur and cur["text"].strip():
            cur["text"] = cur["text"].strip()
            chunks.append(cur)
        cur = None

    for pno, page in enumerate(pages, start=1):
        prev = ""
        for line in page.splitlines():
            clause = detector(line, prev)
            if clause:
                flush()
                cur = {"doc": doc, "clause": clause, "page": pno, "text": line + "\n"}
            else:
                if cur is None:
                    cur = {"doc": doc, "clause": "preamble", "page": pno, "text": ""}
                cur["text"] += line + "\n"
            prev = line
    flush()
    return chunks


def chunk_doc(doc: str, pages: list[str]) -> list[dict]:
    """Split one doc into chunks, trying progressively looser structure.

    1. Numbered/keyword clauses (clean, unambiguous).
    2. If none found, unnumbered titled headings (for prose-style docs).
    3. If still none, page-level chunks so results stay citable.
    """
    chunks = _accumulate(doc, pages, match_heading)
    if any(c["clause"] != "preamble" for c in chunks):
        return chunks

    titled = _accumulate(doc, pages, match_text_heading)
    if any(c["clause"] != "preamble" for c in titled):
        return titled

    return [
        {"doc": doc, "clause": f"page-{pno}", "page": pno, "text": page.strip()}
        for pno, page in enumerate(pages, start=1) if page.strip()
    ]


def make_ids(chunks: list[dict]) -> None:
    """Assign a stable, unique id per chunk (clause ids can repeat in a doc)."""
    seen: dict[tuple, int] = {}
    for c in chunks:
        key = (c["doc"], c["clause"])
        n = seen.get(key, 0)
        seen[key] = n + 1
        c["id"] = f"{c['doc']}#{c['clause']}" + (f"~{n}" if n else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="dir of guideline PDFs")
    ap.add_argument("--out", default=str(Path(__file__).parent / "index.json"))
    args = ap.parse_args()

    docs_dir = Path(args.docs)
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {docs_dir}", file=sys.stderr)
        return 1

    all_chunks: list[dict] = []
    for pdf in pdfs:
        name = pdf.stem
        pages = extract_pages(pdf)
        if not pages:
            continue
        cs = chunk_doc(name, pages)
        make_ids(cs)
        all_chunks.extend(cs)
        print(f"  {name}: {len(pages)} pages -> {len(cs)} chunks")

    index = {
        "source_dir": str(docs_dir),
        "doc_count": len(pdfs),
        "chunk_count": len(all_chunks),
        "chunks": all_chunks,
    }
    Path(args.out).write_text(json.dumps(index, ensure_ascii=False))
    total_chars = sum(len(c["text"]) for c in all_chunks)
    print(f"\nWrote {args.out}")
    print(f"  docs={len(pdfs)} chunks={len(all_chunks)} chars={total_chars}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
