# Results Interpretation

Date: 2026-07-01

## Executive Answer
Yes, this is a real end-to-end build, and it is good as a prototype for the intended thesis.

It successfully implements a Question-Space Driven Knowledge Compilation system (not a plain parser and not a generic chunk-RAG baseline), and it shows strong retrieval-efficiency wins in the current evaluation harness.

## What We Built
The system that ran overnight includes all major layers required by your design:

1. Corpus discovery and inventory over mixed file types.
2. Robust loading with graceful fallback behavior.
3. Source-grounded fragmentation with provenance.
4. Answer-affordance extraction at fragment level with supporting-span validation.
5. Raw-question normalization into canonical question prototypes.
6. Qspace tracking as explicit graph structures.
7. Class discovery as question subspaces (proposed classes).
8. Density and proximity estimation over prototypes.
9. Evidence-role extraction and evidence co-demand.
10. Candidate retrieval-tree generation and explicit cost-based selection.
11. Query-time planner simulation with traversal traces.
12. Baseline comparison against non-Qspace retrieval strategies.
13. Artifact-heavy reporting (CSV, JSONL, plots, trees, markdown, dashboard).

## Scale and Coverage Observed
From the latest run summary:

- Documents inventoried: 559
- Fragments: 2761
- Direct affordance questions: 1023
- Partial affordances: 558
- Question prototypes: 17
- Proposed classes: 5
- Evidence units extracted: 1525
- Retrieval trees compiled: 768
- Simulated traces: 40

This is substantial enough to evaluate system behavior, not just toy behavior.

## Is It Good?
Short answer: good prototype quality, with clear strengths and expected research-grade caveats.

### Strengths
1. The core thesis is operationalized: fragments -> question footprints -> subspaces/classes -> optimized retrieval trees.
2. The retrieval-tree compiler is explicit and inspectable, with candidate-vs-selected scoring.
3. The outputs are rich and auditable, making debugging and analysis practical.
4. The evaluation shows strong efficiency gains without recall collapse.

### Quantitative Signal
From baseline comparison:

- Qspace optimized tree evidence recall: 0.904
- Heading-based tree evidence recall: 0.904
- Fixed chunking evidence recall: 0.904
- Qspace avg context tokens: 59.961
- Heading-based avg context tokens: 141.935
- Fixed chunking avg context tokens: 512.0
- Qspace avg path length: 1.483
- Heading-based avg path length: 3.5
- Qspace avg total cost: 0.151
- Heading-based avg total cost: 0.317

Recorded improvements:

- Context token reduction vs fixed chunking: 88.3%
- Path-length reduction vs heading-based tree: 57.6%

Interpretation: the current pipeline appears to preserve recall while materially reducing retrieval effort proxies.

## Important Caveats (Why This Is Prototype-Good, Not Final-Good)
1. Affordance detection is heuristic fallback (rule-based), not yet LLM-grounded.
2. Density is synthetic, not learned from real query logs.
3. Proximity is heuristic weighted similarity.
4. Class statuses are proposed and provisional.
5. Baseline costs are modeled within this harness, not external production serving metrics.

These caveats are normal for this phase and do not invalidate the prototype value.

## Bottom Line
You built a credible, inspectable research prototype that demonstrates the central claim in measurable terms. It is good enough to move into deeper analysis, error taxonomy, and next-iteration design.

## Where To Cross-Check
- Main summary: outputs/qspace/reports/run_summary.md
- Full narrative: outputs/qspace/reports/final_report.md
- Baseline table: outputs/qspace/evaluation/baseline_comparison.csv
- Prototype table: outputs/qspace/question_prototypes.csv
- Class table: outputs/qspace/classes.csv
- Tree scorecards: outputs/qspace/candidate_tree_scores.csv
- Simulated traces: outputs/qspace/reports/sample_retrieval_traces.md
