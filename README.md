# Qspace

A deterministic, explainable document-intelligence pipeline that converts a corpus into a **question-centric knowledge structure**. Qspace treats documents as containers of answer-bearing fragments, discovers the questions those fragments can answer, organizes the questions into a latent question space, discovers fragment classes as question subspaces, and compiles retrieval-optimized trees that minimize expected answer-path length.

The entire pipeline runs without external APIs and without LLM credentials. Every intermediate result is serialized to inspectable CSV, JSON, or JSONL. Every direct question retains an exact supporting span copied from its source fragment; if the span cannot be validated, the question is dropped.

```
559 documents  ->  2,761 fragments  ->  1,023 direct questions
    ->  17 prototypes  ->  5 classes  ->  768 retrieval trees
    ->  40 query simulations  |  30.21 s  |  0 LLM calls
```

---

## Table of Contents

- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [Retrieval Tree Objective](#retrieval-tree-objective)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Output Artifacts](#output-artifacts)
- [Configuration](#configuration)
- [Limitations](#limitations)
- [License](#license)

---

## Architecture

The pipeline is a linear sequence of 15 stages orchestrated by `qspace_pipeline/run.py`. Each stage reads from the prior stage's output and writes its own artifacts. There are no circular dependencies between stages. The data flow is:

```
Corpus Discovery & Inventory
        |
   Document Loading  (txt, md, html, csv, json, jsonl, pdf*, docx*, xlsx*, pptx*)
        |
   Hierarchical Fragmentation  (sections, paragraphs, list items, table rows)
        |
   Answer-Affordance Detection  (12 intent detectors + span validation)
        |
   Prototype Normalization  (intent + slot signature + evidence-role signature)
        |
   Class Discovery  (agglomerative clustering over intent/role features)
        |
   Density Estimation  (synthetic: support counts + intent risk prior)
        |
   Proximity Estimation  (weighted Jaccard over intent, roles, slots, shapes, fragments, classes)
        |
   Evidence-Role Extraction  &  Co-Demand Matrix
        |
   Qspace Tracker / Graph Construction
        |
   Retrieval Tree Compilation  (5 candidate shapes, scored by J(T), best selected)
        |
   Query Traversal Simulation
        |
   Baseline Evaluation Harness
        |
   Reports, Dashboard, Plots
```

Formats marked with `*` require optional dependencies; the pipeline degrades gracefully when they are absent.

---

## Pipeline Stages

### 1. Corpus Discovery (`corpus_discovery.py`)

Walks the project tree, excluding build/tool directories (`.git`, `__pycache__`, `node_modules`, `venv`, etc.) and scaffolding files (`requirements.txt`, `pyproject.toml`, etc.). Candidate documents are any file with a recognized extension (`.pdf`, `.docx`, `.txt`, `.md`, `.html`, `.csv`, `.json`, `.jsonl`, `.xlsx`, `.pptx`). The directory containing the most documents is selected as the primary corpus. If no documents are found anywhere, a **synthetic multi-domain sample corpus** (policy, finance, SOP) is seeded under `sample_corpus/` so the pipeline can run end-to-end. Every assumption is logged and written to `reports/corpus_discovery.md`.

Domain hints are assigned by keyword matching against path and sampled text (`policy`, `finance`, `sop`, `hr`, `security`).

### 2. Document Loading (`loading.py`)

Each file is dispatched to a format-specific loader:

| Extension | Loader | Notes |
|-----------|--------|-------|
| `.txt` | Plain read | UTF-8 with error tolerance |
| `.md` | Markdown read | Headings extracted as metadata |
| `.html` / `.htm` | `HTMLParser` stripping `<script>`/`<style>` | Falls back to raw text on parse error |
| `.csv` | `csv.reader` | Rendered as `Row: col = val; ...` sentences for downstream detection |
| `.json` | Recursive key-path flattening | `prefix.key = value` lines |
| `.jsonl` | Per-line JSON flattening | Same as JSON, per record |
| `.pdf` | `pypdf` / `PyPDF2` / `pdfplumber` (first available) | Optional; skips with warning if none installed |
| `.docx` | `python-docx` | Optional |
| `.xlsx` | `openpyxl` | Optional; sheet-by-sheet, row-by-row |
| `.pptx` | `python-pptx` | Optional; text frames from each slide |

Loader errors never crash the pipeline. Each document is tagged with its load status (`loaded`, `empty`, `error`, `unsupported`).

### 3. Hierarchical Fragmentation (`fragmentation.py`)

Documents are split into source-grounded fragments with types: `section` (heading lines), `paragraph`, `list_item`, `table_row`. Paragraphs longer than 600 characters are split into sentence windows. Each fragment records its character offsets, parent fragment ID, heading context stack, token estimate, and source path. Fragments shorter than 15 characters (excluding section headers) are dropped.

### 4. Answer-Affordance Detection (`affordances.py`)

Twelve rule-based intent detectors run over every sentence in every fragment:

| Detector | Intent | Key regex signals |
|----------|--------|-------------------|
| `_detect_approval` | `approval_lookup` | `approv` + named approver role |
| `_detect_threshold` | `threshold_lookup` | Dollar amounts + `over`/`under`/`exceed`/`threshold` |
| `_detect_owner` | `owner_lookup` | `owns`/`owner`/`responsible for` + named team |
| `_detect_effective_date` | `effective_date_lookup` | `effective` + date pattern |
| `_detect_applicability` | `applicability_check` | `applies to`/`scope`/`applicab` |
| `_detect_exception` | `exception_lookup` | `except`/`unless`/`prohibited` |
| `_detect_rule` | `rule_lookup` | `must`/`shall`/`require`/`may not` |
| `_detect_metric` | `metric_lookup` | Named metric + dollar/percent value |
| `_detect_variance` | `variance_explanation` | `variance`/`below forecast`/`driven by` |
| `_detect_sla` | `sla_lookup` | `within N hours/days` or `SLA` |
| `_detect_next_step` | `next_step_lookup` | `Step N:` pattern |
| `_detect_escalation` | `escalation_lookup` | `escalated to` + named target |

**Span validation:** every direct question is retained only if the detector's source sentence occurs verbatim (modulo whitespace normalization) in the fragment text. If `_exact_span()` returns `None`, the question is silently dropped. This is the core trust guarantee: no hallucinated supporting evidence.

Broad partial affordances (process-level questions) are also generated from surface cues (`approv`, `Step \d+`, `variance|forecast|revenue`).

### 5. Prototype Normalization (`prototypes.py`)

Raw questions are grouped deterministically by the triple `(intent, slot_signature, evidence_role_signature)`. Each group becomes a **question prototype** with a canonical name like `approval_lookup(region, amount_threshold)`, a canonical question, an answer shape, example raw questions, and the set of supporting fragment/document IDs. Fragment-question edges are built at this stage; each edge carries a support type (`direct`/`partial`), confidence, and supporting span.

### 6. Class Discovery (`classes.py`)

Prototypes are clustered into **question subspaces** (classes) using agglomerative clustering over an explicit binary feature vector: one dimension per intent, one per evidence role. The number of clusters is `max(2, min(5, len(prototypes) // 2))`. If scikit-learn is unavailable, a deterministic fallback groups prototypes by archetype match against three reference sets:

- **Policy-like Fragments** (applicability, rule, exception, approval, owner, effective date, threshold)
- **Finance-like Fragments** (metric, variance, threshold)
- **SOP-like Fragments** (next step, SLA, escalation, owner)

Clusters are named by dominant-intent overlap with these archetypes. All classes are marked `status: proposed`.

### 7. Density Estimation (`proximity.py`)

Question density is a synthetic score (no real query logs) blending:
- Normalized support-fragment count (weight 0.45)
- Inferred importance: 0.5 * intent risk prior + 0.3 * support norm + 0.2 * document norm (weight 0.35)
- Normalized support-document count (weight 0.20)

The intent risk prior is a hand-tuned dictionary (`config.py`) mapping intents like `approval_lookup` (0.90) and `definition_lookup` (0.35) to business/risk weight.

### 8. Proximity Estimation (`proximity.py`)

Pairwise prototype similarity is a weighted sum of six components (weights from `config.py`, summing to 1.0):

| Component | Weight |
|-----------|--------|
| Intent match (0/1) | 0.25 |
| Evidence-role Jaccard | 0.20 |
| Slot Jaccard | 0.20 |
| Answer-shape Jaccard | 0.15 |
| Fragment overlap | 0.10 |
| Class overlap (0/1) | 0.10 |

The result is a symmetric similarity matrix and a proximity network graph (edges at similarity >= 0.45).

### 9. Evidence-Role Extraction (`evidence.py`)

Evidence units are extracted from the `role_values` dictionary produced by each affordance detector. Each unit records the role name, value, source span, confidence (parent confidence * 0.95), and linked prototypes. Duplicate (role, value) pairs within a fragment are deduplicated.

### 10. Evidence Co-Demand Matrix (`evidence.py`)

Co-demand quantifies how often two evidence roles are needed together:

```
S(e_a, e_b) = sum_q  p(q) * I(e_a in q) * I(e_b in q)
```

where `p(q)` is the prototype's normalized density.

### 11. Qspace Tracker (`qspace_tracker.py`)

Constructs a heterogeneous graph with five node types (prototype, fragment, evidence_role, class, document) and five edge types (prototype-fragment, prototype-evidence, prototype-class, fragment-evidence, fragment-document). The graph is serialized to `qspace_graph_nodes.csv` and `qspace_graph_edges.csv`. Prototypes with only one supporting fragment are flagged as thin/coverage-gap regions.

### 12. Retrieval Tree Compilation (`trees.py`)

For each fragment with both evidence and linked prototypes, five candidate retrieval trees are built:

| Candidate | Structure | Character |
|-----------|-----------|-----------|
| `source_order_tree` | Linear chain in document order | Deep, single branch |
| `evidence_role_tree` | root -> role-group -> evidence leaves | Depth 2, grouped by role |
| `question_intent_tree` | root -> intent -> evidence leaves | Grouped by prototype intent |
| `class_objective_tree` | root -> objective -> evidence leaves | Grouped by class objectives |
| `hybrid_optimized_tree` | High-density evidence near root, co-demanded roles grouped | Adaptive based on density + co-demand |

Each candidate is scored and the tree minimizing J(T) is selected. See [Retrieval Tree Objective](#retrieval-tree-objective) for the cost model. Representative trees are rendered as DOT, PNG (via networkx + matplotlib), and interactive HTML.

### 13. Query Traversal Simulation (`planner.py`)

Up to 40 direct-affordance queries are replayed against their compiled trees. For each query, the simulator walks the tree from root to the leaves matching the target evidence roles, recording the traversal path, nodes visited, and context tokens loaded. Results are serialized and a trace report is written to `reports/sample_retrieval_traces.md`.

### 14. Baseline Evaluation (`evaluation.py`)

Silver queries (from validated direct affordances) are scored against five retrieval strategies:

| Strategy | Description |
|----------|-------------|
| `fixed_size_chunking` | 512-token fixed chunks, no tree structure |
| `source_order_fragments` | Linear scan of fragments in document order |
| `heading_based_tree` | Heading -> paragraph -> evidence (depth ~3.5) |
| `generic_fragment_search` | Embedding-style top-k retrieval (modeled, may miss roles) |
| `qspace_optimized_tree` | The pipeline's compiled tree |

Metrics: evidence recall, correct-fragment rate, average nodes visited, average context tokens, average path length, citation locality, and a composite total cost.

### 15. Reporting (`reporting.py`)

Produces three output documents:
- `reports/final_report.md` - Full technical report with tables, class descriptions, tree examples, baseline comparison, limitations, and next steps.
- `reports/run_summary.md` - Headline metrics and improvement percentages.
- `dashboard.html` - Dark-themed HTML dashboard with stat cards, baseline comparison table, artifact links, all plots, and tree visualizations.

---

## Retrieval Tree Objective

The selected tree minimizes a weighted cost:

```
J(T)  =  Total_Cost(T)
       + lambda * tree_complexity(T)
       - mu     * trace_clarity(T)
       - nu     * citation_locality(T)
```

where `Total_Cost` is a weighted sum over these components (weights from `config.py`):

| Component | Weight | Definition |
|-----------|--------|------------|
| Path length | 0.20 | Expected depth to reach answer leaves |
| Nodes visited | 0.10 | Expected distinct nodes traversed |
| Context tokens | 0.20 | Expected tokens loaded during traversal |
| Reasoning steps | 0.10 | Intermediate (non-leaf) nodes visited |
| Ambiguity | 0.10 | Multiple leaves sharing a required role |
| Missing evidence | 0.10 | Required roles absent from the tree |
| Cross-branch jumps | 0.05 | Top-level branches spanned by one query |
| Trace complexity | 0.05 | `max_depth / 6.0` |
| Citation distance | 0.05 | `1.0 - citation_locality` |
| Tree complexity | 0.05 | `(num_nodes + max_depth) / 20.0` |

Regularizers:
- `lambda` = 0.15 (penalizes tree complexity)
- `mu` = 0.20 (rewards trace clarity: `1 / (1 + trace_complexity)`)
- `nu` = 0.15 (rewards citation locality: leaves for the same query clustered together)

All expectations are taken over the question-footprint distribution `p(q|f)`, derived from prototype density.

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/TwinSimLabs/Answer-Aware-parser.git
cd Answer-Aware-parser
pip install -r requirements.txt
```

**Core dependencies** (installed by `requirements.txt`):

- `pandas`, `numpy` -- data manipulation
- `matplotlib` -- plotting (pipeline skips plots gracefully if absent)
- `networkx` -- graph construction and tree visualization
- `scikit-learn` -- agglomerative clustering for class discovery (falls back to intent-archetype grouping if absent)
- `scipy` -- scikit-learn dependency

**Optional loaders** (install individually for format support):

```bash
pip install pypdf        # .pdf text extraction
pip install python-docx  # .docx
pip install openpyxl     # .xlsx
pip install python-pptx  # .pptx
```

The pipeline runs and produces full results without any optional dependencies. Missing loaders result in a logged warning and the affected files being tagged `unsupported`.

---

## Usage

### Run the pipeline

```bash
# Use the current directory as the project root
python run_qspace_pipeline.py

# Or specify a project root explicitly
python run_qspace_pipeline.py /path/to/corpus/project

# Or use the module entry point
python -m qspace_pipeline
```

If the project root contains document files, they are auto-discovered and used as the corpus. If no documents are found, the pipeline seeds a synthetic sample corpus under `sample_corpus/` and runs against it.

### Output

All artifacts are written to `outputs/qspace/`. Start with:

- `dashboard.html` -- visual overview of the entire run
- `reports/run_summary.md` -- headline metrics
- `reports/final_report.md` -- full technical report with tables and analysis

### Sample corpus

The built-in sample corpus covers three domains:

| Domain | Documents | Content |
|--------|-----------|---------|
| Policy | 3 Markdown files + 1 CSV | Procurement, travel/expense, data classification policies; approval matrix |
| Finance | 2 text/Markdown files + 1 JSON | Q2 performance summary, budget guidelines, metric catalog |
| SOP | 3 Markdown/text files | Security incident response, employee onboarding, vendor onboarding |

---

## Repository Structure

```
Answer-Aware-parser/
|-- run_qspace_pipeline.py          # CLI entry point
|-- run_overnight_once.ps1          # Overnight batch runner (pytest + pipeline)
|-- requirements.txt                # Core dependencies
|-- INTEGRATION_EVALUATION_BRIEF.md # Detailed capability and integration brief
|
|-- qspace_pipeline/                # Source code
|   |-- __init__.py                 # Package init, version
|   |-- __main__.py                 # python -m entry point
|   |-- run.py                      # Pipeline orchestrator (15-stage sequence)
|   |-- config.py                   # Paths, cost weights, proximity weights, constants
|   |-- utils.py                    # IO helpers, logging, text tools
|   |-- corpus_discovery.py         # Stage 1: corpus scan, inventory, sample seeding
|   |-- loading.py                  # Stage 2: multi-format document loading
|   |-- fragmentation.py            # Stage 3: hierarchical text fragmentation
|   |-- affordances.py              # Stage 4: heuristic answer-affordance detection
|   |-- prototypes.py               # Stage 5: question prototype normalization
|   |-- classes.py                  # Stage 6: class/question-subspace discovery
|   |-- proximity.py                # Stages 7-8: density + proximity estimation
|   |-- evidence.py                 # Stages 9-10: evidence roles + co-demand matrix
|   |-- qspace_tracker.py           # Stage 11: heterogeneous graph construction
|   |-- trees.py                    # Stage 12: retrieval tree compilation + scoring
|   |-- planner.py                  # Stage 13: query traversal simulation
|   |-- evaluation.py               # Stage 14: baseline comparison harness
|   |-- reporting.py                # Stage 15: reports + HTML dashboard
|   |-- plotting.py                 # matplotlib wrappers (bar, hist, heatmap, scatter)
|   |-- sample_corpus.py            # Synthetic sample corpus generator
|
|-- sample_corpus/                  # Auto-seeded sample documents
|   |-- policies/                   # Procurement, travel, data classification
|   |-- finance/                    # Q2 summary, budget, approval matrix, metric catalog
|   |-- sops/                       # Incident response, onboarding, vendor onboarding
|
|-- outputs/qspace/                 # Pipeline output (gitignored in practice)
    |-- dashboard.html              # Visual HTML dashboard
    |-- run_metadata.json           # Run timestamp, counts, elapsed time
    |-- *.csv / *.json / *.jsonl    # Machine-readable artifacts
    |-- reports/                    # Markdown reports
    |-- plots/                      # PNG visualizations (~25 charts)
    |-- trees/                      # Retrieval tree visuals (DOT, PNG, HTML)
    |-- evaluation/                 # Baseline comparison data
    |-- intermediate/               # Stage-level intermediate files
```

---

## Output Artifacts

### Machine-readable

| File | Contents |
|------|----------|
| `corpus_inventory.csv` | Every discovered file with loader, status, size, domain hints |
| `fragments.csv` | All fragments with type, text, offsets, heading context, token estimate |
| `answer_affordances.csv` | Direct and partial affordances with intent, slots, spans, confidence |
| `question_prototypes.csv` / `.json` | Normalized prototypes with canonical names, slots, roles, support counts |
| `fragment_question_edges.csv` | Fragment-prototype links with support type and confidence |
| `classes.csv` / `.json` | Discovered question subspaces with intents, roles, objectives, members |
| `question_density.csv` | Per-prototype density scores with component breakdown |
| `question_proximity.csv` | Pairwise prototype similarity with component scores |
| `evidence_roles.csv` / `.jsonl` | Extracted evidence units with role, value, span, confidence |
| `evidence_codemand.csv` | Role-pair co-demand matrix |
| `qspace_graph_nodes.csv` | Heterogeneous graph nodes (prototype, fragment, role, class, document) |
| `qspace_graph_edges.csv` | Heterogeneous graph edges with types and weights |
| `retrieval_trees.jsonl` | Full tree records: nodes, scores, candidate comparison, provenance |
| `candidate_tree_scores.csv` | All 5 candidates per fragment with all cost components |
| `selected_tree_scores.csv` | Winning tree per fragment with key metrics |
| `retrieval_simulations.csv` / `.jsonl` | Query traversal traces |
| `evaluation/baseline_comparison.csv` | Strategy comparison: recall, tokens, path length, cost |

### Human-readable

| File | Contents |
|------|----------|
| `dashboard.html` | Dark-themed HTML dashboard with stats, plots, tree visuals |
| `reports/final_report.md` | 16-section technical report |
| `reports/run_summary.md` | Headline metrics and improvement percentages |
| `reports/sample_retrieval_traces.md` | 20 detailed query traversal walkthroughs |
| `reports/corpus_discovery.md` | Assumptions and directory/extension breakdown |
| `plots/*.png` | ~25 charts: fragment distributions, intent frequencies, heatmaps, tree comparisons, proximity networks, bipartite graphs |
| `trees/*.dot` / `.png` / `.html` | Up to 12 representative retrieval trees as graph visualizations |

---

## Configuration

All tunable parameters live in `qspace_pipeline/config.py`:

**Cost weights** (`CostWeights` dataclass): control the retrieval tree objective J(T). Each weight governs one component of the expected cost function. The regularizer weights (`lambda_complexity`, `mu_trace_clarity`, `nu_citation_locality`) balance tree simplicity against retrieval efficiency.

**Proximity weights** (`PROXIMITY_WEIGHTS` dict): control how prototype similarity is computed. Must sum to 1.0.

**Intent risk prior** (`INTENT_RISK_WEIGHT` dict): assigns a business/risk importance to each intent type. Used in density estimation.

**Constants:**
- `MAX_TREE_VISUALS = 12` -- number of representative trees rendered as PNG/DOT/HTML
- `NUM_RETRIEVAL_SIMULATIONS = 40` -- number of query traversals simulated
- `CHARS_PER_TOKEN = 4.0` -- rough tokenizer approximation
- `_MAX_FRAGMENT_CHARS = 600` (in `fragmentation.py`) -- paragraph split threshold

**Excluded directories and files**: `EXCLUDED_DIRS` and `EXCLUDED_FILENAMES` in `config.py` control what the corpus scanner ignores.

---

## Limitations

1. **Heuristic detection.** Answer affordances come from 12 transparent regex/rule detectors. Recall is bounded by the pattern set; novel phrasings, domain-specific language, or non-English text will be missed.

2. **Span validation is necessary but not sufficient.** The pipeline guarantees each direct question quotes real fragment text. It does not guarantee the question is well-posed, correctly interpreted, or complete.

3. **Density is synthetic.** With no real query logs or user interaction data, density is derived from support counts and a hand-tuned intent risk prior. The ranking of prototypes by importance is approximate.

4. **Proximity is a fixed heuristic.** The six-component weighted Jaccard cannot capture semantic similarity beyond exact feature overlap. Two questions about the same concept but with different slot signatures will score as dissimilar.

5. **Class discovery is provisional.** Classes are produced by agglomerative clustering over an explicit binary feature space and are all marked `status: proposed`. The archetype naming heuristic may assign misleading names when a cluster's intent distribution does not match the three reference archetypes.

6. **Tree optimization is local.** Five candidate tree shapes are scored per fragment; the winner minimizes J(T) among those five, not over all possible tree topologies. A global beam search or learned tree construction is a natural extension.

7. **Baseline costs are modeled, not measured.** The evaluation harness simulates retrieval costs with transparent assumptions (fixed token budgets, modeled miss rates) rather than running queries against a live LLM retriever.

8. **No embedding or semantic features.** Clustering, similarity, and affordance detection operate on surface patterns and explicit feature indicators. There is no embedding model and no learned representation.

---

## Author

**Arpit Goel** -- [TwinSimLabs](https://github.com/TwinSimLabs)

---

## License

Not yet specified. Contact the author for licensing inquiries.
