"""Component 3: source-grounded hierarchical fragmentation."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from . import plotting
from .config import Paths
from .utils import (banner, estimate_tokens, log, sub, truncate, write_csv,
                    write_jsonl)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)]|Step\s+\d+:)\s+", re.IGNORECASE)
_ROW_RE = re.compile(r"^Row:\s*(.*)$")
_MAX_FRAGMENT_CHARS = 600


def _new_id(counter: List[int]) -> str:
    counter[0] += 1
    return f"frag_{counter[0]:05d}"


def _make(frag_id: str, doc: Dict, ftype: str, text: str, char_start: int,
          char_end: int, parent: Optional[str], heading_ctx: List[str],
          page: Optional[int] = None, table_id: Optional[str] = None) -> Dict:
    return {
        "fragment_id": frag_id,
        "document_id": doc["document_id"],
        "source_path": doc["file_path"],
        "parent_fragment_id": parent,
        "fragment_type": ftype,
        "text": text.strip(),
        "char_start": char_start,
        "char_end": char_end,
        "page_number": page,
        "table_id": table_id,
        "heading_context": list(heading_ctx),
        "token_estimate": estimate_tokens(text),
        "domain_hints": doc.get("domain_hints", []),
        "title": doc.get("title", ""),
    }


def _split_paragraph(text: str) -> List[str]:
    """Split an over-long paragraph into sentence windows."""
    if len(text) <= _MAX_FRAGMENT_CHARS:
        return [text]
    from .utils import split_sentences
    sents = split_sentences(text)
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 > _MAX_FRAGMENT_CHARS and cur:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        chunks.append(cur.strip())
    return chunks or [text]


def _fragment_document(doc: Dict, counter: List[int]) -> List[Dict]:
    text = doc["text"]
    if not text.strip():
        return []
    frags: List[Dict] = []
    heading_stack: List[str] = []
    section_parent: Optional[str] = None
    is_structured = doc["extension"] in (".csv", ".json", ".jsonl", ".xlsx")

    lines = text.splitlines(keepends=True)
    offset = 0
    para_buf: List[str] = []
    para_start = 0

    def flush_paragraph(end: int) -> None:
        nonlocal para_buf, para_start
        raw = "".join(para_buf).strip()
        para_buf = []
        if not raw:
            return
        for chunk in _split_paragraph(raw):
            fid = _new_id(counter)
            ftype = "list_item" if _LIST_RE.match(chunk) else (
                "table_row" if _ROW_RE.match(chunk) else "paragraph")
            frags.append(_make(fid, doc, ftype, chunk, para_start,
                               para_start + len(chunk), section_parent,
                               heading_stack))

    for ln in lines:
        stripped = ln.strip()
        m = _HEADING_RE.match(ln.rstrip())
        if m:
            flush_paragraph(offset)
            level = len(m.group(1))
            title = m.group(2).strip()
            heading_stack = heading_stack[: level - 1] + [title]
            fid = _new_id(counter)
            frag = _make(fid, doc, "section", title, offset,
                         offset + len(ln), None, heading_stack)
            frags.append(frag)
            section_parent = fid
            offset += len(ln)
            para_start = offset
            continue
        if not stripped:
            flush_paragraph(offset)
            offset += len(ln)
            para_start = offset + 0
            continue
        if _LIST_RE.match(stripped) or _ROW_RE.match(stripped):
            # each list/table row is its own fragment
            flush_paragraph(offset)
            fid = _new_id(counter)
            ftype = "table_row" if _ROW_RE.match(stripped) else "list_item"
            tbl = f"{doc['document_id']}_table" if ftype == "table_row" else None
            frags.append(_make(fid, doc, ftype, stripped, offset,
                               offset + len(ln), section_parent, heading_stack,
                               table_id=tbl))
            offset += len(ln)
            para_start = offset
            continue
        if not para_buf:
            para_start = offset
        para_buf.append(ln)
        offset += len(ln)
    flush_paragraph(offset)

    # Drop trivial fragments (headings kept even if short).
    cleaned = [f for f in frags if f["fragment_type"] == "section"
               or len(f["text"]) >= 15]
    return cleaned


def fragment_documents(paths: Paths, documents: List[Dict]) -> List[Dict]:
    banner("FRAGMENT", "Fragmentation")
    counter = [0]
    all_frags: List[Dict] = []
    per_doc: Counter[str] = Counter()
    for doc in documents:
        frags = _fragment_document(doc, counter)
        for f in frags:
            per_doc[doc["document_id"]] += 1
        all_frags.extend(frags)

    write_jsonl(paths.intermediate / "fragments.jsonl", all_frags)
    write_csv(paths.out / "fragments.csv", all_frags, columns=[
        "fragment_id", "document_id", "source_path", "parent_fragment_id",
        "fragment_type", "text", "char_start", "char_end", "page_number",
        "table_id", "heading_context", "token_estimate"])

    type_counts: Counter[str] = Counter(f["fragment_type"] for f in all_frags)
    log("FRAGMENT", f"Documents processed: {len(documents)}")
    log("FRAGMENT", f"Total fragments: {len(all_frags)}")
    log("FRAGMENT", "Fragment types:")
    for t, c in type_counts.most_common():
        sub(f"{t}: {c}")
    n_docs_with = len([d for d in documents if per_doc[d['document_id']]])
    avg = len(all_frags) / max(1, n_docs_with)
    log("FRAGMENT", f"Average fragments per document: {avg:.1f}")
    log("FRAGMENT", "Sample fragments:")
    for f in all_frags[:4]:
        sub(f"{f['fragment_id']} [{f['fragment_type']}]: "
            f"\"{truncate(f['text'], 90)}\"")

    plotting.bar(paths.plots / "fragment_type_counts.png",
                 list(type_counts.keys()), list(type_counts.values()),
                 "Fragment type counts", ylabel="fragments")
    plotting.bar(paths.plots / "fragments_per_document.png",
                 [d for d, _ in per_doc.most_common(20)],
                 [c for _, c in per_doc.most_common(20)],
                 "Fragments per document (top 20)", ylabel="fragments",
                 horizontal=True)
    plotting.hist(paths.plots / "fragment_length_distribution.png",
                  [len(f["text"]) for f in all_frags],
                  "Fragment length distribution", "characters")
    log("FRAGMENT", f"Saved: {paths.intermediate / 'fragments.jsonl'}")
    return all_frags
