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

try:
    import pdfplumber  # optional: structured parsing of curriculum tables
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

_ROMAN_TO_INT = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
_INT_TO_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}

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
        # -layout preserves table columns/rows (curriculum docs are tables) while
        # still reading prose sensibly.
        paged = subprocess.run(
            ["pdftotext", "-layout", "-q", str(pdf_path), "-"],
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


# Year-Semester token used in curriculum tables, e.g. "III-I" (year 3, sem 1).
_SEM_TOKEN = re.compile(r"\b(V|IV|III|II|I)-(II|I)\b")
_TERM_BANNER = re.compile(r"^\s*(monsoon|spring)\s*$", re.IGNORECASE)


def is_curriculum(pages: list[str]) -> bool:
    """A semester-wise course table: has several distinct year-sem tokens."""
    text = "\n".join(pages)
    tokens = {m.group(0) for m in _SEM_TOKEN.finditer(text)}
    return len(tokens) >= 3


def chunk_curriculum(doc: str, pages: list[str]) -> list[dict]:
    """Chunk a curriculum table into one chunk per semester block.

    Blocks are delimited by Monsoon/Spring term banners; each block is labelled
    with the year-semester token found inside it (e.g. clause 'III-I'), so a
    semester's course list is a single citable chunk.
    """
    chunks: list[dict] = []
    cur = None
    n = 0

    def flush():
        nonlocal cur
        if cur and cur["text"].strip():
            cur["text"] = cur["text"].strip()
            m = _SEM_TOKEN.search(cur["text"])
            cur["clause"] = m.group(0) if m else cur["clause"]
            chunks.append(cur)
        cur = None

    for pno, page in enumerate(pages, start=1):
        for line in page.splitlines():
            if _TERM_BANNER.match(line):
                flush()
                n += 1
                cur = {"doc": doc, "clause": f"sem-{n}", "page": pno,
                       "text": line.strip() + "\n"}
            else:
                if cur is None:
                    cur = {"doc": doc, "clause": "overview", "page": pno, "text": ""}
                cur["text"] += line + "\n"
    flush()

    non_overview = [c for c in chunks if c["clause"] != "overview"]
    if not non_overview:  # no term banners → fall back to generic pipeline
        return chunk_doc(doc, pages, _allow_curriculum=False)
    return chunks


def _advance_sem(year: int, sem: int) -> tuple[int, int]:
    return (year, 2) if sem == 1 else (year + 1, 1)


def _credits_of(cell: str) -> int:
    """Sum the numbers in a credit-distribution cell, e.g. '- 4 - - - -' -> 4."""
    return sum(int(x) for x in re.findall(r"\d+", cell or ""))


def parse_curriculum_tables(pdf_path: Path, doc: str) -> list[dict]:
    """Parse a semester-wise curriculum PDF into one structured chunk per
    semester using pdfplumber, so course credits are unambiguous to the reader.

    Returns [] if pdfplumber is unavailable or finds no usable tables (caller
    then falls back to the text-based chunker).
    """
    if not HAS_PDFPLUMBER:
        return []
    chunks: list[dict] = []
    year = sem = 0
    try:
        pdf = pdfplumber.open(pdf_path)
    except Exception:  # noqa: BLE001
        return []
    with pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables() or []:
                rows = table
                i = 0
                while i < len(rows):
                    row = [(c or "").strip() for c in rows[i]]
                    joined = " ".join(row)
                    term = next((c for c in row if c.lower() in ("monsoon", "spring")), None)
                    if not term:
                        i += 1
                        continue
                    tok = _SEM_TOKEN.search(joined)
                    if tok:
                        year = _ROMAN_TO_INT[tok.group(1)]
                        sem = _ROMAN_TO_INT[tok.group(2)]
                    elif (year, sem) == (0, 0):
                        year, sem = 1, 1
                    else:
                        year, sem = _advance_sem(year, sem)
                    label = f"{_INT_TO_ROMAN[year]}-{_INT_TO_ROMAN[sem]}"
                    course_row = rows[i + 1] if i + 1 < len(rows) else None
                    text = _format_semester(course_row, term, label)
                    if text:
                        chunks.append({"doc": doc, "clause": label, "page": pno, "text": text})
                    i += 2
    return chunks


def _is_course_name(s: str) -> bool:
    s = s.strip()
    if not s or s.lower() in ("sub total", "total", "monsoon", "spring"):
        return False
    if re.match(r"^[A-Z]{1,3}\d", s):        # a course code
        return False
    if s in ("Full", "Half") or re.match(r"^[\d\s\-]+$", s):  # type / numbers
        return False
    return any(ch.isalpha() for ch in s)


def _format_semester(course_row, term: str, label: str) -> str:
    """Render one semester's courses as an explicit, column-labelled table.

    Anchors on the course-NAME column (the only reliably complete one — many
    electives have no code), and aligns type / L-T-P / credit columns by row
    index. Codes are shown only when their count matches the course count, since
    blank-code rows otherwise misalign that column.
    """
    if not course_row:
        return ""
    col = [(c or "").split("\n") for c in course_row]

    def find(pred):
        return next((c for c in col if c and pred(c[0].strip())), [])

    # Names = the column with the most course-name-like entries.
    names_col = max(col, key=lambda c: sum(_is_course_name(x) for x in c), default=[])
    names = [x.strip() for x in names_col if _is_course_name(x)]
    if not names:
        return ""
    types = find(lambda s: s in ("Full", "Half"))
    ltp = find(lambda s: re.match(r"^\d\s+\d\s+\d", s))
    credits = next((c for c in col
                    if c and re.match(r"^[-\d](\s|$)", c[0].strip()) and c is not ltp), [])
    codes = find(lambda s: re.match(r"^[A-Z]{1,3}\d", s))
    code_vals = [x.strip() for x in codes if x.strip()]
    use_codes = len(code_vals) == len(names)

    def g(lst, i):
        return lst[i].strip() if i < len(lst) else ""

    lines = [f"{term} — Semester {label}", "Course | Full/Half | L-T-P | Credits"]
    total = 0
    for i, name in enumerate(names):
        cr = _credits_of(g(credits, i))
        total += cr
        code = f"{code_vals[i]} " if use_codes else ""
        lines.append(f"{code}{name} | {g(types, i)} | {g(ltp, i)} | {cr} credits")
    lines.append(f"Total credits: {total}")
    return "\n".join(lines)


def chunk_pdf(pdf_path: Path, doc: str, pages: list[str]) -> list[dict]:
    """Top-level dispatch: structured curriculum parsing, else text chunking."""
    if is_curriculum(pages):
        structured = parse_curriculum_tables(pdf_path, doc)
        if structured:
            return structured
        return chunk_curriculum(doc, pages)  # text fallback
    return chunk_doc(doc, pages, _allow_curriculum=False)


def chunk_doc(doc: str, pages: list[str], _allow_curriculum: bool = True) -> list[dict]:
    """Split one doc into chunks, trying progressively looser structure.

    0. Curriculum tables → one chunk per semester block.
    1. Numbered/keyword clauses (clean, unambiguous).
    2. If none found, unnumbered titled headings (for prose-style docs).
    3. If still none, page-level chunks so results stay citable.
    """
    if _allow_curriculum and is_curriculum(pages):
        return chunk_curriculum(doc, pages)

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


MAX_CHUNK_CHARS = 1500  # oversized chunks bury content + blur embeddings


def _split_text(text: str, max_chars: int) -> list[str]:
    """Greedily pack paragraphs (then sentences) into parts <= max_chars."""
    blocks = re.split(r"\n\s*\n", text)
    if len(blocks) == 1:
        blocks = text.split("\n")
    parts: list[str] = []
    cur = ""
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if len(b) > max_chars:  # a single huge block -> split on sentences
            for s in re.split(r"(?<=[.;])\s+", b):
                if cur and len(cur) + len(s) > max_chars:
                    parts.append(cur.strip())
                    cur = ""
                cur += s + " "
            continue
        if cur and len(cur) + len(b) > max_chars:
            parts.append(cur.strip())
            cur = ""
        cur += b + "\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


def split_oversized(chunks: list[dict], max_chars: int = MAX_CHUNK_CHARS) -> list[dict]:
    """Split any chunk longer than max_chars into smaller same-clause parts."""
    out: list[dict] = []
    for c in chunks:
        if len(c["text"]) <= max_chars:
            out.append(c)
            continue
        for part in _split_text(c["text"], max_chars):
            out.append({**c, "text": part})
    return out


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
        cs = chunk_pdf(pdf, name, pages)
        cs = split_oversized(cs)
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
