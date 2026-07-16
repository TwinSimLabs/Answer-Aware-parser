"""Central configuration: output paths, tunable weights, and constants."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# ----------------------------------------------------------------------------
# Directories that are never treated as corpus sources.
# ----------------------------------------------------------------------------
EXCLUDED_DIRS = {
    ".git", "__pycache__", "node_modules", "venv", ".venv", "env",
    "dist", "build", "outputs", "target", ".next", ".idea", ".vscode",
    ".mypy_cache", ".pytest_cache", "qspace_pipeline", ".ipynb_checkpoints",
    "site-packages",
}

# Document extensions we consider part of a candidate corpus.
DOCUMENT_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".html", ".htm",
    ".csv", ".xlsx", ".pptx", ".json", ".jsonl",
}

# Project-scaffolding files that are never treated as corpus documents even
# though they share a document extension.
EXCLUDED_FILENAMES = {
    "requirements.txt", "requirements-dev.txt", "dev-requirements.txt",
    "setup.cfg", "pyproject.toml", "package.json", "package-lock.json",
    "tsconfig.json", "license", "license.txt", "license.md", "notice",
    "changelog.md", "contributing.md", "code_of_conduct.md",
    ".gitignore", "manifest.json", "run_metadata.json",
}

# Rough characters-per-token estimate for the token approximations.
CHARS_PER_TOKEN = 4.0


@dataclass
class Paths:
    """All output paths, rooted at ``<project>/outputs/qspace``."""

    project_root: Path
    out: Path = field(init=False)
    intermediate: Path = field(init=False)
    plots: Path = field(init=False)
    reports: Path = field(init=False)
    trees: Path = field(init=False)
    evaluation: Path = field(init=False)
    sample_corpus: Path = field(init=False)

    def __post_init__(self) -> None:
        self.out = self.project_root / "outputs" / "qspace"
        self.intermediate = self.out / "intermediate"
        self.plots = self.out / "plots"
        self.reports = self.out / "reports"
        self.trees = self.out / "trees"
        self.evaluation = self.out / "evaluation"
        self.sample_corpus = self.project_root / "sample_corpus"

    def ensure(self) -> None:
        for p in (self.out, self.intermediate, self.plots, self.reports,
                  self.trees, self.evaluation):
            p.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Cost-function weights for the retrieval tree objective J(T).
# ----------------------------------------------------------------------------
@dataclass
class CostWeights:
    path_length: float = 0.20
    nodes_visited: float = 0.10
    context_tokens: float = 0.20
    reasoning_steps: float = 0.10
    ambiguity: float = 0.10
    missing_evidence: float = 0.10
    cross_branch: float = 0.05
    trace_complexity: float = 0.05
    citation_distance: float = 0.05
    tree_complexity: float = 0.05
    # J(T) regularizers
    lambda_complexity: float = 0.15   # + tree complexity penalty
    mu_trace_clarity: float = 0.20    # - trace clarity reward
    nu_citation_locality: float = 0.15  # - citation locality reward


# ----------------------------------------------------------------------------
# Proximity similarity weights (must sum to 1.0).
# ----------------------------------------------------------------------------
PROXIMITY_WEIGHTS: Dict[str, float] = {
    "intent": 0.25,
    "evidence_role_jaccard": 0.20,
    "slot_jaccard": 0.20,
    "answer_shape_jaccard": 0.15,
    "fragment_overlap": 0.10,
    "class_overlap": 0.10,
}

# ----------------------------------------------------------------------------
# Intent -> business/risk weight prior (used for density estimation).
# ----------------------------------------------------------------------------
INTENT_RISK_WEIGHT: Dict[str, float] = {
    "approval_lookup": 0.90,
    "threshold_lookup": 0.85,
    "exception_lookup": 0.80,
    "escalation_lookup": 0.78,
    "rule_lookup": 0.70,
    "applicability_check": 0.65,
    "sla_lookup": 0.62,
    "variance_explanation": 0.60,
    "owner_lookup": 0.55,
    "next_step_lookup": 0.52,
    "metric_lookup": 0.50,
    "effective_date_lookup": 0.45,
    "input_output_lookup": 0.42,
    "definition_lookup": 0.35,
}
DEFAULT_RISK_WEIGHT = 0.40

# Number of representative fragments to render as tree PNG/DOT/HTML artifacts.
MAX_TREE_VISUALS = 12
# Number of retrieval traces to simulate and print.
NUM_RETRIEVAL_SIMULATIONS = 40
