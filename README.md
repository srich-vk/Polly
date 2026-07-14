# Local Policy-Doc Query CLI

Ask natural-language questions across the college's scattered policy/guideline
PDFs (in `College Guidelines/`) and get a precise answer **with an exact
citation** — document name + clause/section + page. Runs **fully local**: a
local LLM via [ollama] for reasoning, and stdlib Python for everything else. No
cloud, no external endpoints, no pip dependencies.

## How it works

Rather than embedding-based RAG, this uses **agentic lexical search** — better
suited to policy queries that hinge on exact terms (clause numbers, dates, room
codes):

1. `build_index.py` extracts every PDF and splits it on **structural
   boundaries** (clause/section numbers) into `index.json`. The 7 scanned
   image-only PDFs (incl. `Hostel_Rules_2025`) are OCR'd automatically so the
   whole corpus is searchable.
2. `retrieval.py` is a pure-stdlib **BM25** engine over those chunks.
3. `policy_ask.py` gives a local model two tools — `search_docs` and
   `read_section` — and lets it search, read the exact clause, and **reformulate
   on a miss**. Every answer is grounded in retrieved text and cited; if the
   answer isn't in the docs, it says so instead of guessing. A **citation
   verifier** flags any cited clause that doesn't actually exist in the index.

## Prerequisites

- Python 3 (stdlib only)
- [ollama] running locally with the model pulled:
  ```bash
  ollama pull qwen2.5:3b        # ~1.9 GB, sized for a 4 GB GPU
  ```
- To *rebuild* the index only: `poppler` (`pdftotext`, `pdftoppm`) and
  `tesseract` OCR. Not needed to just query — `index.json` is committed.

## Usage

```bash
python3 policy_ask.py "can a student below 5.5 CGPA take 20 credits?"
```
Or install the `policy-ask` shortcut on your PATH:
```bash
cat > ~/.local/bin/policy-ask <<'EOF'
#!/usr/bin/env bash
exec python3 "/home/srich-vk/Desktop/IIIT RAG/policy_ask.py" "$@"
EOF
chmod +x ~/.local/bin/policy-ask
```

| Command | What it does |
|---|---|
| `policy-ask "question"` | Agentic LLM answer with citations |
| `policy-ask` | Interactive REPL (`/search`, `/read`, `/list`, `/quit`) |
| `policy-ask --search "kw" [-n N] [--full]` | Raw BM25 hits, **no LLM** (instant, exact-term) |
| `policy-ask read <doc> [clause]` | Print a full section verbatim |
| `policy-ask --list` | List all 63 documents |
| `policy-ask --show-work` | Show each search/read the model runs |

**Environment overrides:** `POLICY_MODEL` (default `qwen2.5:3b`), `OLLAMA_HOST`
(default `http://127.0.0.1:11434`). Also `--model` / `--host` flags.

## Rebuilding the index

Run whenever the PDFs in `College Guidelines/` change:

```bash
python3 build_index.py --docs "College Guidelines" --out index.json
```

## Files

| File | Role |
|---|---|
| `build_index.py` | PDF → clause-chunked `index.json` (with OCR fallback) |
| `index.json` | Generated index — 63 docs, 631 chunks |
| `retrieval.py` | Stdlib BM25 retrieval engine |
| `policy_ask.py` | Agent loop + tool dispatch + citation verifier + CLI |

## Curriculum / course-list queries

Semester-wise curriculum PDFs (e.g. `BTech-ECD-V2`) are extracted with layout
preserved and chunked **one chunk per semester**, labelled with the doc's
year-sem notation (`III-I`). You can query in plain arabic notation — a
normalizer maps `3-1` → `III-I` automatically:

```bash
policy-ask "what courses do I take in ECD sem 3-1?"
policy-ask --search "CSE 2-2"        # -> BTech-CSE-V2, clause II-II
```

## Notes & limitations

- Curriculum tables are dense; the local 3B model reliably lists the **courses**
  for a semester but can misread the L/T/P/credit **columns**. The citation
  (e.g. `III-I`) lets you verify against the source.
- The LLM adds convenience (natural-language questions, interpreting
  conditional/numeric policy) at a small residual hallucination risk that the
  `--search` path avoids entirely. Use `--search` for pure exact-term lookups.
- If the model narrates a search instead of running it, the loop nudges it to
  actually call the tool rather than returning the narration.

[ollama]: https://ollama.com
