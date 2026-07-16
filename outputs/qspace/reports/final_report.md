# Question-Space Driven Knowledge Compilation — Final Report

_Generated: 2026-07-01T23:09:04_

This prototype treats documents as containers of answer-bearing fragments, discovers the questions those fragments can answer, organizes them into a latent question space, discovers classes as question subspaces, and compiles retrieval-optimized trees that minimize expected answer-path length.

## 1. Corpus summary

- Project root: `C:\Work\Answer-Aware-parser`
- Selected corpus directory: `C:\Work\Answer-Aware-parser\sample_corpus`
- Seeded synthetic sample corpus: **False**
- Total documents inventoried: **559**
- Extensions: {'.md': 7, '.json': 275, '.pdf': 274, '.txt': 2, '.csv': 1}

**Assumptions:**

- Selected 'sample_corpus' as the primary corpus directory because it contains the most candidate documents (559). All discovered documents across the project are still inventoried.

## 2. Fragmentation summary

- Total fragments: **2761**
- Fragment types: {'paragraph': 2700, 'list_item': 13, 'section': 40, 'table_row': 8}

## 3. Answer affordance summary

- Detection method: **heuristic rule-based fallback**
- Fragments with affordances: **1006**
- Direct questions: **1023** | Partial questions: **558**
- Validation: every direct question retains an exact supporting span found in its fragment; unverifiable spans were dropped.

## 4. Top question prototypes

| Prototype | Intent | Fragments | Docs | Density |
| --- | --- | --- | --- | --- |
| threshold_lookup(amount_threshold) | threshold_lookup | 234 | 123 | 0.9737 |
| metric_lookup(metric) | metric_lookup | 212 | 105 | 0.8208 |
| rule_lookup(subject) | rule_lookup | 154 | 73 | 0.648 |
| exception_lookup(condition) | exception_lookup | 99 | 66 | 0.5197 |
| variance_explanation(metric) | variance_explanation | 85 | 37 | 0.3878 |
| applicability_check(audience) | applicability_check | 63 | 38 | 0.3466 |
| effective_date_lookup(document) | effective_date_lookup | 48 | 35 | 0.2694 |
| threshold_lookup(amount_threshold, region) | threshold_lookup | 39 | 33 | 0.3137 |
| metric_lookup(metric, period) | metric_lookup | 17 | 6 | 0.141 |
| variance_explanation(metric) | variance_explanation | 15 | 9 | 0.1603 |
| owner_lookup(object) | owner_lookup | 10 | 8 | 0.1375 |
| sla_lookup(step) | sla_lookup | 9 | 6 | 0.143 |
| approval_lookup(region, transaction_type) | approval_lookup | 8 | 1 | 0.1787 |
| approval_lookup(transaction_type) | approval_lookup | 6 | 4 | 0.1805 |
| approval_lookup(amount_threshold, transaction_type) | approval_lookup | 5 | 4 | 0.1781 |

## 5. Qspace tracker summary

- Node counts: {'prototypes': 17, 'fragments': 1006, 'evidence_roles': 17, 'classes': 5, 'documents': 234}
- Edge counts: {'prototype_fragment': 1023, 'prototype_evidence': 29, 'prototype_class': 17, 'fragment_evidence': 1525, 'fragment_document': 768}
- Thin/coverage-gap prototypes: 1

## 6. Proposed classes / question subspaces

### class_001 — Policy-like Fragments (status: proposed)
- Dominant intents: rule_lookup, exception_lookup, applicability_check, effective_date_lookup, owner_lookup
- Common evidence roles: condition, rule, exception, audience, scope, effective_date
- Class objectives: applicability_check, rule_lookup, exception_lookup, approval_lookup, owner_lookup, effective_date_lookup
- Members: 7 prototypes, 332 fragments across 142 documents

### class_002 — Approval Lookup Subspace (status: proposed)
- Dominant intents: approval_lookup
- Common evidence roles: approver, region, amount_threshold
- Class objectives: approval_lookup
- Members: 4 prototypes, 19 fragments across 9 documents

### class_003 — Metric Lookup Subspace (status: proposed)
- Dominant intents: metric_lookup
- Common evidence roles: metric, value, period
- Class objectives: metric_lookup
- Members: 2 prototypes, 226 fragments across 107 documents

### class_004 — Variance Explanation Subspace (status: proposed)
- Dominant intents: variance_explanation
- Common evidence roles: variance, variance_driver
- Class objectives: variance_explanation
- Members: 2 prototypes, 99 fragments across 41 documents

### class_005 — Threshold Lookup Subspace (status: proposed)
- Dominant intents: threshold_lookup
- Common evidence roles: amount_threshold, region
- Class objectives: threshold_lookup
- Members: 2 prototypes, 273 fragments across 136 documents

## 7. Question density analysis

Density blends supporting-fragment count, supporting-document count, an intent risk prior, and synthetic frequency (no real query logs are available). Top prototypes by density:

| Prototype | Density |
| --- | --- |
| threshold_lookup(amount_threshold) | 0.9737 |
| metric_lookup(metric) | 0.8208 |
| rule_lookup(subject) | 0.648 |
| exception_lookup(condition) | 0.5197 |
| variance_explanation(metric) | 0.3878 |
| applicability_check(audience) | 0.3466 |
| threshold_lookup(amount_threshold, region) | 0.3137 |
| effective_date_lookup(document) | 0.2694 |

## 8. Question proximity analysis

Proximity is an inspectable weighted blend of intent, evidence-role Jaccard, slot Jaccard, answer-shape Jaccard, fragment overlap, and class overlap. Closest prototype pairs:

| A | B | Similarity |
| --- | --- | --- |
| variance_explanation(metric) | variance_explanation(metric) | 0.801 |
| approval_lookup(amount_threshold, transaction_type) | approval_lookup(amount_threshold, region, transaction_type) | 0.7867 |
| approval_lookup(region, transaction_type) | approval_lookup(amount_threshold, region, transaction_type) | 0.7667 |
| metric_lookup(metric) | metric_lookup(metric, period) | 0.7347 |
| threshold_lookup(amount_threshold) | threshold_lookup(amount_threshold, region) | 0.7 |
| approval_lookup(region, transaction_type) | approval_lookup(transaction_type) | 0.7 |
| approval_lookup(transaction_type) | approval_lookup(amount_threshold, transaction_type) | 0.7 |
| approval_lookup(region, transaction_type) | approval_lookup(amount_threshold, transaction_type) | 0.6333 |

## 9. Evidence role analysis

- Evidence units extracted: **1525**
- Distinct roles: 17
- Strong co-demand pairs are visualized in `plots/evidence_codemand_heatmap.png`.

## 10. Retrieval tree compilation results

- Trees compiled: **768**
- Selected type distribution: {'question_intent_tree': 284, 'hybrid_optimized_tree': 480, 'class_objective_tree': 4}
- Each fragment is compiled into 5 candidate trees (source_order, evidence_role, question_intent, class_objective, hybrid_optimized); the tree minimizing J(T) is selected.

## 11. Candidate tree comparison (mean total_cost)

| Candidate type | Mean total_cost |
| --- | --- |
| source_order_tree | 0.1952 |
| evidence_role_tree | 0.2698 |
| question_intent_tree | 0.2321 |
| class_objective_tree | 0.2392 |
| hybrid_optimized_tree | 0.1831 |

## 12. Sample optimized trees

### tree_frag_00004_v1 (question_intent_tree)
- optimized_for: metric_lookup
- expected_path_length=2.0, expected_context_tokens=20.0, trace_clarity=0.75, total_cost=0.2217, J=-0.0333
- visual: `trees/tree_frag_00004.png` / `.html` / `.dot`

### tree_frag_00005_v1 (hybrid_optimized_tree)
- optimized_for: rule_lookup
- expected_path_length=1.0, expected_context_tokens=28.0, trace_clarity=0.857, total_cost=0.1788, J=-0.1201
- visual: `trees/tree_frag_00005.png` / `.html` / `.dot`

### tree_frag_00006_v1 (question_intent_tree)
- optimized_for: metric_lookup
- expected_path_length=2.0, expected_context_tokens=21.0, trace_clarity=0.75, total_cost=0.2227, J=-0.0323
- visual: `trees/tree_frag_00006.png` / `.html` / `.dot`

## 13. Sample retrieval traces

See `reports/sample_retrieval_traces.md` for full traces. 40 query traversals were simulated.

## 14. Baseline comparison

| Strategy | Evid. recall | Avg tokens | Avg path | Avg cost |
| --- | --- | --- | --- | --- |
| fixed_size_chunking | 0.904 | 512.0 | N/A | 0.738 |
| source_order_fragments | 0.904 | 7327.325 | 3.699 | 6.076 |
| heading_based_tree | 0.904 | 141.935 | 3.5 | 0.317 |
| generic_fragment_search | 0.664 | 135.572 | 2.0 | 0.309 |
| qspace_optimized_tree | 0.904 | 59.961 | 1.483 | 0.151 |

**Improvements:**

- context_token_reduction_vs_fixed_pct: 88.3%
- path_length_reduction_vs_heading_pct: 57.6%

## 15. Known limitations

- **Heuristic detection.** Answer affordances come from transparent regex/rule detectors, not an LLM. Recall is bounded by the pattern set; novel phrasings may be missed.
- **Span validation** guarantees each direct question quotes real fragment text, but does not guarantee the question is well-posed.
- **Generated questions may be incomplete**; partial affordances flag missing evidence but are not exhaustively enumerated.
- **Class discovery is provisional** (status = proposed) and produced by clustering an explicit intent/evidence feature space.
- **Density is synthetic**, derived from support counts and an intent risk prior; there are no real query logs.
- **Proximity is heuristic**, a fixed weighted similarity.
- **Tree optimization is heuristic**: five candidate structures are scored and the best J(T) is chosen; it is not a global optimum over all possible trees.
- **Baseline costs are modeled** with transparent assumptions, not measured against a live LLM retriever.

## 16. Next recommended improvements

- Plug in an LLM for affordance detection and prototype naming, keeping the span-validation gate.
- Learn density from real query logs / click data.
- Replace the fixed proximity weights with a learned metric.
- Add a global tree search (e.g. cost-guided beam search) beyond the five templates.
- Add embeddings as an auxiliary clustering/visualization aid.
- Expand evaluation to a live retrieval harness with answer grading.

