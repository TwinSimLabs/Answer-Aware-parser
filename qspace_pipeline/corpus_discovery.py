"""Component 0 & 1: corpus discovery and inventory."""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from . import plotting, sample_corpus
from .config import DOCUMENT_EXTENSIONS, EXCLUDED_DIRS, EXCLUDED_FILENAMES, Paths
from .utils import (banner, estimate_tokens, log, sub, truncate, write_csv,
                    write_json)

# Domain hint keywords -> label.
_DOMAIN_HINTS: Dict[str, List[str]] = {
    "policy": ["policy", "approval", "compliance", "governance", "classification"],
    "finance": ["budget", "revenue", "financial", "forecast", "metric", "variance"],
    "sop": ["sop", "procedure", "onboarding", "incident", "escalation", "step"],
    "hr": ["employee", "hiring", "people", "hr"],
    "security": ["security", "encryption", "access", "risk"],
}


def _iter_candidate_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs in-place
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS
                       and not d.startswith(".")]
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext in DOCUMENT_EXTENSIONS and name.lower() not in EXCLUDED_FILENAMES:
                files.append(Path(dirpath) / name)
    return files


def _guess_domain(path: Path, sample_text: str) -> List[str]:
    hay = (str(path).lower() + " " + sample_text[:400].lower())
    hits = []
    for domain, kws in _DOMAIN_HINTS.items():
        if any(kw in hay for kw in kws):
            hits.append(domain)
    return hits or ["general"]


def _title_guess(path: Path, sample_text: str) -> str:
    for line in sample_text.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            return truncate(s, 80)
    return path.stem.replace("_", " ").title()


def _quick_read(path: Path, limit: int = 2000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except Exception:
        return ""


def discover_and_inventory(paths: Paths) -> Tuple[List[Dict], Dict]:
    """Scan project, (optionally) seed a sample corpus, and build inventory."""
    banner("SCAN", "Corpus discovery")
    root = paths.project_root
    log("SCAN", f"Project root: {root}")

    files = _iter_candidate_files(root)
    assumptions: List[str] = []
    seeded = False

    if not files:
        log("SCAN", "No document files detected anywhere in the project.")
        log("SCAN", "ASSUMPTION: seeding a synthetic multi-domain sample corpus "
                    "so the pipeline can run end-to-end.")
        created = sample_corpus.generate(paths.sample_corpus)
        log("SCAN", f"Seeded {len(created)} sample documents at "
                    f"{paths.sample_corpus}")
        assumptions.append(
            "No pre-existing corpus was found in the project. A synthetic "
            "multi-domain sample corpus (policy / finance / SOP) was generated "
            "under ./sample_corpus and used as the corpus.")
        seeded = True
        files = _iter_candidate_files(root)

    # Group candidate files by their top-level directory under root.
    dir_counts: Counter[str] = Counter()
    for f in files:
        try:
            rel = f.relative_to(root)
            top = rel.parts[0] if len(rel.parts) > 1 else "."
        except ValueError:
            top = str(f.parent)
        dir_counts[top] += 1

    candidate_dirs = [d for d, _ in dir_counts.most_common()]
    log("SCAN", f"Candidate corpus directories found: {candidate_dirs}")

    # Select the directory containing the most documents as the primary corpus.
    if seeded:
        selected_dir = paths.sample_corpus
    else:
        top_dir = dir_counts.most_common(1)[0][0]
        selected_dir = root if top_dir == "." else root / top_dir
        assumptions.append(
            f"Selected '{top_dir}' as the primary corpus directory because it "
            f"contains the most candidate documents ({dir_counts[top_dir]}). "
            f"All discovered documents across the project are still inventoried.")
    log("SCAN", f"Selected corpus directory: {selected_dir}")

    ext_counts: Counter[str] = Counter(f.suffix.lower() for f in files)
    log("SCAN", "File counts by extension:")
    for ext, c in sorted(ext_counts.items(), key=lambda kv: -kv[1]):
        sub(f"{ext:<8} {c}")
    log("SCAN", f"Total candidate documents: {len(files)}")
    log("SCAN", "Assumptions:")
    for a in assumptions:
        sub(f"- {a}")

    # Build inventory rows.
    rows: List[Dict] = []
    for i, f in enumerate(sorted(files), start=1):
        try:
            stat = f.stat()
            size = stat.st_size
            mtime = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        except OSError:
            size, mtime = 0, ""
        sample = _quick_read(f)
        char_count = len(sample)  # refined later at load time
        loader = _LOADER_BY_EXT.get(f.suffix.lower(), "unsupported")
        rows.append({
            "document_id": f"doc_{i:04d}",
            "file_path": str(f),
            "file_name": f.name,
            "extension": f.suffix.lower(),
            "file_size_bytes": size,
            "modified_time": mtime,
            "detected_loader": loader,
            "load_status": "pending",
            "character_count": char_count,
            "token_estimate": estimate_tokens(sample),
            "document_title_guess": _title_guess(f, sample),
            "possible_domain_hints": _guess_domain(f, sample),
        })

    # Persist inventory.
    write_csv(paths.out / "corpus_inventory.csv", rows)
    write_json(paths.out / "corpus_inventory.json", rows)

    _inventory_plots(paths, rows, dir_counts)
    _write_discovery_report(paths, root, selected_dir, candidate_dirs,
                            ext_counts, len(files), assumptions, seeded)

    banner("INVENTORY", "Corpus inventory")
    lengths = [r["character_count"] for r in rows]
    log("INVENTORY", f"Inventoried {len(rows)} candidate documents.")
    if lengths:
        med = sorted(lengths)[len(lengths) // 2]
        longest = max(rows, key=lambda r: r["character_count"])
        log("INVENTORY", f"Median sampled length: {med:,} chars")
        log("INVENTORY", f"Longest sampled document: {longest['file_name']}")
    log("INVENTORY", f"Saved: {paths.out / 'corpus_inventory.csv'}")

    meta = {
        "project_root": str(root),
        "selected_corpus_dir": str(selected_dir),
        "candidate_dirs": candidate_dirs,
        "extension_counts": dict(ext_counts),
        "total_documents": len(files),
        "assumptions": assumptions,
        "seeded_sample_corpus": seeded,
    }
    return rows, meta


# Loader dispatch table shared with the loading module.
_LOADER_BY_EXT = {
    ".txt": "text", ".md": "markdown", ".csv": "csv", ".json": "json",
    ".jsonl": "jsonl", ".html": "html", ".htm": "html", ".pdf": "pdf",
    ".docx": "docx", ".xlsx": "xlsx", ".pptx": "pptx",
}


def _inventory_plots(paths: Paths, rows: List[Dict], dir_counts) -> None:
    ext_counter: Counter[str] = Counter(r["extension"] for r in rows)
    plotting.bar(
        paths.plots / "file_type_distribution.png",
        list(ext_counter.keys()), list(ext_counter.values()),
        "File type distribution", xlabel="extension", ylabel="documents")
    plotting.hist(
        paths.plots / "document_length_distribution.png",
        [r["character_count"] for r in rows],
        "Document length distribution (sampled chars)", "characters")
    top_dirs = dir_counts.most_common(15)
    plotting.bar(
        paths.plots / "top_directories_by_doc_count.png",
        [d for d, _ in top_dirs], [c for _, c in top_dirs],
        "Top directories by document count", ylabel="documents",
        horizontal=True)


def _write_discovery_report(paths: Paths, root: Path, selected: Path,
                            candidate_dirs, ext_counts, total, assumptions,
                            seeded) -> None:
    lines = ["# Corpus Discovery Report", "",
             f"- Project root: `{root}`",
             f"- Selected corpus directory: `{selected}`",
             f"- Seeded synthetic sample corpus: **{seeded}**",
             f"- Total candidate documents: **{total}**", "",
             "## Candidate directories", ""]
    lines += [f"- `{d}`" for d in candidate_dirs] or ["- (none)"]
    lines += ["", "## File counts by extension", "",
              "| Extension | Count |", "| --- | --- |"]
    for ext, c in sorted(ext_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{ext}` | {c} |")
    lines += ["", "## Assumptions", ""]
    lines += [f"- {a}" for a in assumptions] or ["- None."]
    (paths.reports / "corpus_discovery.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
