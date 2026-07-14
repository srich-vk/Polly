#!/usr/bin/env python3
"""
policy_ask.py — Local, agentic policy-doc query CLI.

Answers questions across the college guideline PDFs by giving a local LLM
(via ollama) two retrieval tools — search_docs and read_section — and letting
it search, read, and reformulate on a miss. Every answer is grounded in the
retrieved text and cited; if the answer isn't in the docs, it says so.

Pure stdlib + ollama. No pip dependencies.

    policy-ask "can a student below 5.5 CGPA take 20 credits?"
    policy-ask --search "hostel fee"      # raw BM25, no LLM
    policy-ask read Overload_Registration # dump a section verbatim
    policy-ask --list                     # list all documents
    policy-ask                            # interactive REPL
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

from retrieval import get_retriever


def _c(code: str, text: str, stream=sys.stdout) -> str:
    """Wrap text in an ANSI code only when writing to a terminal."""
    return f"\033[{code}m{text}\033[0m" if stream.isatty() else text

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.environ.get("POLICY_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
NUM_CTX = 8192
MAX_ITERS = 6
TIMEOUT = 180

SYSTEM_PROMPT = """You are a college policy assistant. You answer questions about \
institute guidelines using ONLY the local policy documents, retrieved through the \
search_docs and read_section tools.

How to answer:
1. Call search_docs with the most specific terms from the question (exact clause \
numbers, dates, room codes, policy names where known).
2. If the top results do not clearly contain the answer, REFORMULATE and search again \
(synonyms, broader terms, a likely document name). Do not give up after one search.
3. Call read_section to read the full verbatim text of the relevant clause before \
answering. Base your answer on that text, not the snippet alone.

Hard rules (non-negotiable):
- Answer ONLY from retrieved document text. Never use outside knowledge.
- CITE every answer with the document name and clause/section number and page, e.g. \
"(Overload_Registration_Guidelines-July24, clause 3.2, p2)". Quote exact wording for \
anything numeric or conditional.
- If the documents do not contain the answer, say so explicitly, e.g. "This is not \
covered in the available policy documents." Do NOT guess. A confidently wrong policy \
answer is worse than admitting the information is not found.
- NEVER announce or describe a search in prose (do not write "I will now search..." or \
"let me look for..."). Just call the tool directly. Write prose ONLY for your final \
answer, once you already have the information or have confirmed it is absent.
"""

# Phrases that mean the model narrated a next step instead of calling the tool.
_CONTINUATION_HINTS = (
    "i will now", "i'll now", "will now conduct", "conduct another", "another search",
    "search again", "let me search", "let's search", "let me look", "let's try",
    "let us try", "let me refine", "let's refine", "i will search", "i'll search",
    "i will conduct", "i'll conduct", "i will look", "i'll look", "i am going to",
    "try searching", "refine our search", "refine the search", "next, i",
    "please wait", "i need to read", "i need to call", "i will call", "i'll call",
    "need to read the", "i need to check", "i should read", "let me read",
)
MAX_NUDGES = 3


def _looks_incomplete(content: str) -> bool:
    low = content.lower()
    if any(h in low for h in _CONTINUATION_HINTS):
        return True
    # Names a tool but didn't actually call it (a promise, not an answer).
    if ("read_section" in low or "search_docs" in low) and any(
        w in low for w in ("call", "function", "need to", "wait", "will")
    ):
        return True
    return False


# Small models sometimes emit a tool call as text instead of via the structured
# tool-call field, in either JSON form  -> read_section {"doc": "X"}
# or key=value form                     -> read_section doc="X" clause="Y"
# Recover both and run them.
_JSON_CALL_RE = re.compile(r"(search_docs|read_section)\s*[\(:]?\s*(\{[^{}]*\})")
_KV_CALL_RE = re.compile(
    r"(search_docs|read_section)\s*\(?\s*"
    r"((?:\w+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s,)]+)[\s,]*)+)"
)
_KV_PAIR_RE = re.compile(r"(\w+)\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s,)]+)")


def _extract_text_calls(content: str) -> list[tuple[str, dict]]:
    content = content or ""
    calls: list[tuple[str, dict]] = []
    seen: set[str] = set()

    def add(name: str, args: dict):
        key = name + repr(sorted(args.items()))
        if key not in seen and args:
            seen.add(key)
            calls.append((name, args))

    for m in _JSON_CALL_RE.finditer(content):
        try:
            add(m.group(1), json.loads(m.group(2)))
        except json.JSONDecodeError:
            continue
    for m in _KV_CALL_RE.finditer(content):
        args = {km.group(1): km.group(2).strip("\"'")
                for km in _KV_PAIR_RE.finditer(m.group(2))}
        add(m.group(1), args)
    return calls

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Keyword search across all college policy/guideline documents (BM25). "
                "Returns top matching sections with document name, clause/section "
                "number, page and a snippet. Use exact terms where known."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Search terms."},
                    "limit": {"type": "integer", "description": "Max results (default 6)."},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_section",
            "description": (
                "Read the full verbatim text of a policy section. Give the document "
                "name (partial ok) and optionally a clause/section number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc": {"type": "string", "description": "Document name or part of it."},
                    "clause": {"type": "string", "description": "Clause/section number, e.g. '3.2'."},
                },
                "required": ["doc"],
            },
        },
    },
]


# --------------------------------------------------------------------------- #
# Tool dispatch (backed by retrieval.py)
# --------------------------------------------------------------------------- #
def _embed_query(text: str) -> "list[float] | None":
    """Embed a query for hybrid search; return None on any failure (BM25-only)."""
    try:
        payload = json.dumps(
            {"model": EMBED_MODEL, "input": ["search_query: " + text]}
        ).encode()
        req = urllib.request.Request(
            f"{HOST}/api/embed", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["embeddings"][0]
    except Exception:  # noqa: BLE001
        return None


def _tool_search_docs(args: dict) -> str:
    r = get_retriever()
    kw = str(args.get("keyword", ""))
    k = int(args.get("limit", 6) or 6)
    qvec = _embed_query(kw) if r.has_embeddings else None
    hits = r.search_hybrid(kw, qvec, k)
    if not hits:
        return (
            f'No sections matched "{args.get("keyword", "")}". Try different terms, '
            f"an exact clause number, or a document name."
        )
    body = "\n\n".join(
        f'[{h["doc"]} · clause {h["clause"]} · p{h["page"]}] (score {h["score"]})\n{h["snippet"]}'
        for h in hits
    )
    return (
        body
        + "\n\nThese are truncated snippets. Before answering, call read_section with a "
        "result's document and clause to get its full text (e.g. complete course lists, "
        "totals, exact wording)."
    )


def _tool_read_section(args: dict) -> str:
    r = get_retriever()
    hits = r.read_section(str(args.get("doc", "")), args.get("clause"))
    if not hits:
        return f'No section found for document "{args.get("doc", "")}".'
    return "\n\n---\n\n".join(
        f'### {c["doc"]} · clause {c["clause"]} · page {c["page"]}\n{c["text"]}' for c in hits
    )


DISPATCH = {"search_docs": _tool_search_docs, "read_section": _tool_read_section}


# --------------------------------------------------------------------------- #
# ollama client (stdlib)
# --------------------------------------------------------------------------- #
class OllamaError(RuntimeError):
    pass


def _chat(messages: list[dict]) -> dict:
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
        "options": {"num_ctx": NUM_CTX, "temperature": 0.1},
    }).encode()
    req = urllib.request.Request(
        f"{HOST}/api/chat", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise OllamaError(
            f"Cannot reach ollama at {HOST} ({e}). Is `ollama serve` running?"
        ) from e
    if "error" in body:
        err = body["error"]
        if "not found" in err.lower():
            raise OllamaError(f"Model '{MODEL}' not available. Run: ollama pull {MODEL}")
        raise OllamaError(err)
    return body["message"]


def ask(question: str, show_work: bool = False) -> str:
    """Run the agentic loop and return the final cited answer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    nudges = 0
    tool_used = False
    for _ in range(MAX_ITERS):
        msg = _chat(messages)
        messages.append(msg)
        content = (msg.get("content") or "").strip()
        # Structured tool calls, plus any the model wrote as plain text.
        norm: list[tuple[str, dict]] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw = fn.get("arguments", {})
            norm.append((fn.get("name", ""), json.loads(raw) if isinstance(raw, str) else raw))
        if not norm:
            norm = _extract_text_calls(content)
        if norm:
            tool_used = True
            for name, args in norm:
                handler = DISPATCH.get(name)
                result = handler(args) if handler else f"Unknown tool: {name}"
                if show_work:
                    print("  " + _c("2", f"↳ {name}({json.dumps(args)})", sys.stderr), file=sys.stderr)
                messages.append({"role": "tool", "content": result})
            continue
        # Never let the model conclude (especially "not found") without searching:
        # small models sometimes answer from their own empty knowledge. Force at
        # least one search first.
        if not tool_used and nudges < MAX_NUDGES:
            nudges += 1
            if show_work:
                print("  " + _c("2", "↳ (nudge: concluded without searching)", sys.stderr),
                      file=sys.stderr)
            messages.append({
                "role": "user",
                "content": (
                    "You have not searched the documents yet. Call search_docs now with "
                    "the key terms from the question before drawing any conclusion."
                ),
            })
            continue

        # The model wrote prose but made no tool call. If it merely announced a
        # search instead of performing it, nudge it to actually call the tool
        # rather than accepting the narration as a (non-)answer.
        if _looks_incomplete(content) and nudges < MAX_NUDGES:
            nudges += 1
            if show_work:
                print("  " + _c("2", "↳ (nudge: narrated a search but didn't call it)", sys.stderr),
                      file=sys.stderr)
            messages.append({
                "role": "user",
                "content": (
                    "Do not describe your next step in prose. Call search_docs or "
                    "read_section now if you need more information. If you already have "
                    "enough, give the final answer with citations. If the documents do "
                    "not contain the answer, say so explicitly."
                ),
            })
            continue
        return _verify_citations(content)
    return _verify_citations(
        "I could not find this in the available documents after several searches. "
        "Try rephrasing with the exact terms used in the policy."
    )


# --------------------------------------------------------------------------- #
# Citation verifier — flags cited (doc, clause) pairs that don't exist
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", s.lower())


def _verify_citations(answer: str) -> str:
    if not answer:
        return answer
    r = get_retriever()
    docs = r.list_docs()
    ans_norm = _norm(answer)
    cited = [d for d in docs if _norm(d) in ans_norm]
    _NOT_FOUND = (
        "not covered", "not found", "not available", "not contain", "no information",
        "none of the", "cannot find", "could not find", "not mention", "not specif",
        "not addressed", "unable to find", "no relevant", "not in the available",
        "not present", "do not contain", "does not contain", "isn't covered", "no mention",
        "no sections", "no matching", "there are no", "matching the term", "not able to find",
    )
    low = answer.lower()
    looks_not_found = any(m in low for m in _NOT_FOUND)

    warnings: list[str] = []
    if not cited and not looks_not_found:
        warnings.append("answer contains no recognizable document citation")

    # For each cited doc, verify clause numbers mentioned near it actually exist.
    for d in cited:
        for m in re.finditer(re.escape(d).replace(r"\_", r"[\s_]"), answer, re.I):
            tail = answer[m.end(): m.end() + 60]
            cm = re.search(r"clause\s*([\w.\-]+)|section\s*([\w.\-]+)", tail, re.I)
            if cm:
                clause = cm.group(1) or cm.group(2)
                if not r.citation_exists(d, clause):
                    warnings.append(f'cited "{d}" clause {clause} not found in index')

    if warnings:
        note = "\n".join(f"  - {w}" for w in dict.fromkeys(warnings))
        answer += "\n\n" + _c("33", "⚠ citation check:") + f"\n{note}"
    return answer


# --------------------------------------------------------------------------- #
# Non-LLM helpers
# --------------------------------------------------------------------------- #
def print_search(query: str, limit: int, full: bool, doc: str | None = None) -> None:
    r = get_retriever()
    hits = r.search(query, limit)
    if doc:
        hits = [h for h in hits if doc.lower() in h["doc"].lower()]
    if not hits:
        print(f'No matches for "{query}".')
        return
    for i, h in enumerate(hits, 1):
        print(f'{i}. [{h["doc"]} · clause {h["clause"]} · p{h["page"]}]  (score {h["score"]})')
        if full:
            secs = r.read_section(h["doc"], h["clause"])
            print("   " + (secs[0]["text"] if secs else h["snippet"]).replace("\n", "\n   "))
        else:
            print(f'   {h["snippet"]}')
        print()


def print_read(doc: str, clause: str | None) -> None:
    r = get_retriever()
    hits = r.read_section(doc, clause)
    if not hits:
        print(f'No section found for "{doc}"' + (f' clause {clause}' if clause else "") + ".")
        return
    for c in hits:
        print(f'### {c["doc"]} · clause {c["clause"]} · page {c["page"]}')
        print(c["text"])
        print()


def repl() -> None:
    print("Policy assistant (local). Ask a question, or:")
    print("  /search <kw>   raw search    /read <doc> [clause]   /list   /quit\n")
    while True:
        try:
            line = input("policy> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/quit", "/exit", ":q"):
            return
        if line == "/list":
            print("\n".join(get_retriever().list_docs()))
        elif line.startswith("/search "):
            print_search(line[8:].strip(), 6, False)
        elif line.startswith("/read "):
            parts = line[6:].split()
            print_read(parts[0], parts[1] if len(parts) > 1 else None)
        else:
            try:
                print(ask(line, show_work=True) + "\n")
            except OllamaError as e:
                print(f"error: {e}\n", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="policy-ask", description="Query college policy documents locally."
    )
    p.add_argument("query", nargs="*", help="Question, or 'read <doc> [clause]'.")
    p.add_argument("-s", "--search", action="store_true", help="Raw BM25 search, no LLM.")
    p.add_argument("-n", "--limit", type=int, default=6, help="Max search results.")
    p.add_argument("--full", action="store_true", help="Show full clause text in search.")
    p.add_argument("--doc", help="Restrict --search to docs matching this name.")
    p.add_argument("--list", action="store_true", help="List all documents.")
    p.add_argument("--show-work", action="store_true", help="Print the model's tool calls.")
    p.add_argument("--model", help="Override model (default $POLICY_MODEL or qwen2.5:3b).")
    p.add_argument("--host", help="Override ollama host (default $OLLAMA_HOST).")
    args = p.parse_args(argv)

    global MODEL, HOST
    if args.model:
        MODEL = args.model
    if args.host:
        HOST = args.host.rstrip("/")

    if args.list:
        print("\n".join(get_retriever().list_docs()))
        return 0

    tokens = args.query
    if tokens and tokens[0] == "read":
        if len(tokens) < 2:
            p.error("read requires a document name")
        print_read(tokens[1], tokens[2] if len(tokens) > 2 else None)
        return 0

    if args.search:
        if not tokens:
            p.error("--search requires a query")
        print_search(" ".join(tokens), args.limit, args.full, args.doc)
        return 0

    if not tokens:
        repl()
        return 0

    try:
        print(ask(" ".join(tokens), show_work=args.show_work))
    except OllamaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
