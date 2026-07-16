# Answer-Aware Parser (Qspace Pipeline) - Integration Evaluation Brief

## 1) Executive Summary

This project is a deterministic, explainable document intelligence pipeline that converts a corpus into a question-centric knowledge structure called Qspace.

In plain terms, it:
- Finds and loads documents from a project folder.
- Splits documents into grounded text fragments.
- Detects what questions each fragment can answer (rule-based, no LLM required).
- Normalizes those questions into reusable prototypes.
- Clusters prototypes into emergent classes (question subspaces).
- Extracts evidence roles (for example: approver, threshold, region, SLA).
- Builds and scores retrieval trees per fragment.
- Simulates retrieval and compares Qspace retrieval against baselines.
- Produces machine-readable artifacts and human-readable reports/dashboard.

Current design emphasizes transparency and traceability over model sophistication.

## 2) What It Does (Capabilities)

### Core capabilities
- Corpus auto-discovery with inventorying and metadata profiling.
- Multi-format loading: txt, md, html, csv, json, jsonl, plus optional pdf/docx/xlsx/pptx.
- Hierarchical fragmentation into sections, paragraphs, list/table rows.
- Heuristic answer-affordance extraction with span validation.
- Prototype normalization by intent + slot signature + evidence-role signature.
- Class discovery via clustering over explicit intent/role features.
- Question density scoring (synthetic) and pairwise proximity matrix.
- Evidence role extraction and evidence co-demand matrix.
- Qspace graph construction (nodes/edges for prototypes/fragments/roles/classes/docs).
- Retrieval-tree generation (5 candidate shapes) and objective-based selection.
- Query traversal simulation.
- Baseline evaluation harness (fixed chunking, heading tree, generic search, etc.).
- Reporting outputs: markdown reports, CSV/JSON artifacts, plots, HTML dashboard.

### Operational capabilities
- Works without external APIs and without LLM credentials.
- Fails gracefully for missing optional loaders.
- Produces reproducible outputs from deterministic rules and fixed formulas.

## 3) How It Works (Pipeline Mechanics)

Execution entrypoints:
- `python run_qspace_pipeline.py [PROJECT_ROOT]`
- `python -m qspace_pipeline`

Main orchestrator sequence:
1. Discover corpus and generate inventory.
2. Load documents into normalized text representations.
3. Fragment text into source-grounded units.
4. Detect direct/partial answer affordances per fragment.
5. Normalize to question prototypes and build fragment-question edges.
6. Discover classes (question subspaces) from prototype features.
7. Compute prototype density and pairwise proximity.
8. Extract evidence roles and compute co-demand.
9. Build Qspace tracker graph.
10. Compile and score retrieval-tree candidates per fragment; select best objective J(T).
11. Simulate retrieval traces.
12. Evaluate against baseline strategies.
13. Emit reports and dashboard.

### Retrieval tree objective
The selected tree minimizes a weighted objective J(T) over expected retrieval cost terms (path length, nodes visited, tokens, ambiguity, missing evidence, branch jumps, complexity, citation locality/clarity).

### Explainability model
Every direct question is retained only if it has an exact supporting span in source text. This is a key trust feature for compliance/audit workflows.

## 4) Inputs, Outputs, and Integration Surfaces

### Inputs expected
- Project directory containing document files.
- Supported extensions include `.pdf`, `.docx`, `.txt`, `.md`, `.html`, `.csv`, `.xlsx`, `.pptx`, `.json`, `.jsonl`.

### Major outputs (under `outputs/qspace/`)
- Structure and semantics:
  - `question_prototypes.csv/.json`
  - `classes.csv/.json`
  - `qspace_graph_nodes.csv`
  - `qspace_graph_edges.csv`
  - `qspace_tracker.json`
- Evidence and retrieval:
  - `evidence_roles.csv/.jsonl`
  - `evidence_codemand.csv`
  - `retrieval_trees.jsonl`
  - `selected_tree_scores.csv`
  - `candidate_tree_scores.csv`
  - `retrieval_simulations.csv/.jsonl`
- Evaluation and reporting:
  - `evaluation/baseline_comparison.csv`
  - `reports/final_report.md`
  - `reports/run_summary.md`
  - `dashboard.html`
  - `run_metadata.json`

### Practical integration points
- Consume CSV/JSONL outputs in your own retrieval, analytics, or RAG orchestration layer.
- Use `retrieval_trees.jsonl` + `evidence_roles.csv` as interpretable retrieval plans.
- Use `question_prototypes.csv` and `classes.csv` to seed intent routing/taxonomy.
- Use `baseline_comparison.csv` as a regression metric source in CI.

## 5) Measured Behavior in Current Workspace

From existing `outputs/qspace/run_metadata.json`:
- Documents: 559
- Fragments: 2761
- Direct questions: 1023
- Partial questions: 558
- Prototypes: 17
- Classes: 5
- Evidence units: 1525
- Trees: 768
- Simulations: 40
- Elapsed: 30.21s
- LLM usage: false

Interpretation:
- The pipeline is already handling moderately large corpora and producing dense downstream artifacts.
- Runtime profile appears suitable for batch/offline indexing workflows.

## 6) Strengths for Integration

- Deterministic and auditable pipeline (compliance-friendly).
- No mandatory dependency on paid model APIs.
- Rich artifact graph enables multiple downstream use-cases.
- Span-grounded direct questions reduce hallucinated provenance.
- Baseline comparison built in, enabling evidence-based adoption decisions.

## 7) Risks and Limitations

- Affordance detection is heuristic regex/rule based, so recall and linguistic coverage are limited.
- Density and proximity are synthetic/heuristic, not learned from production traffic.
- Retrieval optimization searches among five template tree families, not all possible trees.
- Baseline evaluation uses modeled assumptions rather than live user traffic and answer graders.
- Domain-specific detectors may need adaptation for domains beyond policy/finance/SOP style corpora.

## 8) Integration Fit Checklist (Chatbot-Readable)

Use this as a quick decision rubric.

### Strong fit if most are true
- You need explainable, source-grounded retrieval structures.
- You prefer deterministic offline indexing over opaque end-to-end LLM extraction.
- You want CSV/JSON artifacts to feed an existing orchestration stack.
- Your corpus is primarily procedural, policy, governance, or operational text.
- You can tolerate heuristic extraction quality initially and improve iteratively.

### Weak fit if most are true
- You require high semantic recall on unconstrained natural language from day one.
- You need multilingual, domain-agnostic extraction without detector extension work.
- You need real-time low-latency per-request indexing (this is batch-oriented).
- You require model-native generative reasoning quality as the primary retrieval mechanism.

## 9) Recommendation

Recommended as an integration candidate for projects that value provenance, deterministic behavior, and inspectable retrieval planning.

Suggested adoption strategy:
1. Integrate as an offline indexing/preprocessing stage first.
2. Consume `question_prototypes`, `classes`, `evidence_roles`, and `retrieval_trees` in your existing runtime.
3. Run side-by-side with your current retriever and compare using `evaluation/baseline_comparison.csv`.
4. If needed, incrementally replace or augment heuristic affordance detection with an LLM stage while preserving span-validation safeguards.

## 10) Minimal Integration Contract

If another project wants to integrate quickly, start with these files:
- `outputs/qspace/question_prototypes.csv`
- `outputs/qspace/classes.csv`
- `outputs/qspace/evidence_roles.csv`
- `outputs/qspace/retrieval_trees.jsonl`
- `outputs/qspace/evaluation/baseline_comparison.csv`

This gives enough structure for intent routing, retrieval planning, evidence grounding, and measurable comparison.
