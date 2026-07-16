"""Shared utilities: logging, IO helpers, and small text tools."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .config import CHARS_PER_TOKEN

# ----------------------------------------------------------------------------
# Terminal logging with consistent [SECTION] tags.
# ----------------------------------------------------------------------------
_SECTION_WIDTH = 70


def banner(section: str, title: str = "") -> None:
    line = "=" * _SECTION_WIDTH
    print("\n" + line)
    print(f"[{section}] {title}".rstrip())
    print(line)


def log(section: str, message: str = "") -> None:
    print(f"[{section}] {message}")


def sub(message: str = "", indent: int = 4) -> None:
    print(" " * indent + message)


# ----------------------------------------------------------------------------
# IO helpers.
# ----------------------------------------------------------------------------
def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]],
              columns: Sequence[str] | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Still create an (almost) empty file with a header if columns known.
        with path.open("w", encoding="utf-8", newline="") as fh:
            if columns:
                csv.DictWriter(fh, fieldnames=list(columns)).writeheader()
        return 0
    if columns is None:
        cols: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in cols:
                    cols.append(k)
        columns = cols
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _csv_cell(r.get(k, "")) for k in columns})
    return len(rows)


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple, set)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


# ----------------------------------------------------------------------------
# Text helpers.
# ----------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(round(len(text) / CHARS_PER_TOKEN)))


_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def slug(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s[:max_len] or "x"


def truncate(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "\u2026"
