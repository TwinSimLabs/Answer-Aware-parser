"""Plotting helpers. All plots degrade gracefully if data is missing."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from .utils import log

try:  # matplotlib is optional at import time.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _HAVE_MPL = True
except Exception as exc:  # pragma: no cover
    _HAVE_MPL = False
    log("PLOT", f"matplotlib unavailable ({exc}); plots will be skipped.")


def available() -> bool:
    return _HAVE_MPL


def _save(fig, path: Path) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _guard(path: Path, reason: str) -> None:
    log("PLOT", f"skipped {path.name}: {reason}")


def bar(path: Path, labels: Sequence[str], values: Sequence[float], title: str,
        xlabel: str = "", ylabel: str = "count", rotate: int = 45,
        color: str = "#4C72B0", horizontal: bool = False) -> None:
    if not _HAVE_MPL:
        return
    if not labels or not values:
        _guard(path, "no data")
        return
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), 4.5))
    if horizontal:
        ax.barh(range(len(labels)), values, color=color)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
    else:
        ax.bar(range(len(labels)), values, color=color)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=rotate, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    ax.set_title(title)
    _save(fig, path)


def hist(path: Path, values: Sequence[float], title: str, xlabel: str,
         bins: int = 30, color: str = "#55A868") -> None:
    if not _HAVE_MPL:
        return
    vals = [v for v in values if v is not None]
    if not vals:
        _guard(path, "no data")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(vals, bins=min(bins, max(3, len(set(vals)))), color=color,
            edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frequency")
    _save(fig, path)


def grouped_bar(path: Path, groups: Sequence[str], series: Dict[str, Sequence[float]],
                title: str, ylabel: str) -> None:
    if not _HAVE_MPL:
        return
    if not groups or not series:
        _guard(path, "no data")
        return
    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 1.2), 4.8))
    n = len(series)
    width = 0.8 / max(1, n)
    x = np.arange(len(groups))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + i * width, vals, width, label=name)
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, path)


def heatmap(path: Path, matrix: Sequence[Sequence[float]], row_labels: Sequence[str],
            col_labels: Sequence[str], title: str, cmap: str = "viridis",
            annotate: bool = False) -> None:
    if not _HAVE_MPL:
        return
    if not matrix or not row_labels or not col_labels:
        _guard(path, "no data")
        return
    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 0.6 + 2),
                                    max(4, len(row_labels) * 0.5 + 1.5)))
    im = ax.imshow(arr, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if annotate and arr.size <= 400:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center",
                        color="white" if arr[i, j] < arr.max() * 0.6 else "black",
                        fontsize=6)
    _save(fig, path)


def scatter(path: Path, xs, ys, labels, title, xlabel="x", ylabel="y",
            colors=None) -> None:
    if not _HAVE_MPL:
        return
    if not xs:
        _guard(path, "no data")
        return
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(xs, ys, c=colors if colors is not None else "#4C72B0",
               cmap="tab10", s=40, alpha=0.8, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save(fig, path)
