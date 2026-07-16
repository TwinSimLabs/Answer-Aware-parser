"""Component 15: reporting, run summary, and HTML dashboard."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .config import Paths


def _md_table(headers: List[str], rows: List[List]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def write_reports(paths: Paths, ctx: Dict) -> None:
    _final_report(paths, ctx)
    _run_summary(paths, ctx)
    _dashboard(paths, ctx)


def _final_report(paths: Paths, ctx: Dict) -> None:
    meta = ctx["corpus_meta"]
    prototypes = ctx["prototypes"]
    classes = ctx["classes"]
    density = ctx["density"]
    proto_by_id = {p["prototype_id"]: p for p in prototypes}
    eval_summary = ctx["eval"]["summary"]
    improvements = ctx["eval"]["improvements"]
    trees = ctx["trees"]

    L: List[str] = []
    L += [f"# Question-Space Driven Knowledge Compilation — Final Report", "",
          f"_Generated: {datetime.now().isoformat(timespec='seconds')}_", "",
          "This prototype treats documents as containers of answer-bearing "
          "fragments, discovers the questions those fragments can answer, "
          "organizes them into a latent question space, discovers classes as "
          "question subspaces, and compiles retrieval-optimized trees that "
          "minimize expected answer-path length.", ""]

    L += ["## 1. Corpus summary", "",
          f"- Project root: `{meta['project_root']}`",
          f"- Selected corpus directory: `{meta['selected_corpus_dir']}`",
          f"- Seeded synthetic sample corpus: **{meta['seeded_sample_corpus']}**",
          f"- Total documents inventoried: **{meta['total_documents']}**",
          f"- Extensions: {meta['extension_counts']}", "",
          "**Assumptions:**", ""]
    L += [f"- {a}" for a in meta["assumptions"]] or ["- None."]
    L.append("")

    L += ["## 2. Fragmentation summary", "",
          f"- Total fragments: **{ctx['num_fragments']}**",
          f"- Fragment types: {ctx['fragment_type_counts']}", ""]

    L += ["## 3. Answer affordance summary", "",
          f"- Detection method: **{'LLM' if ctx['used_llm'] else 'heuristic rule-based fallback'}**",
          f"- Fragments with affordances: **{ctx['num_affordance_frags']}**",
          f"- Direct questions: **{ctx['num_direct']}** | "
          f"Partial questions: **{ctx['num_partial']}**",
          "- Validation: every direct question retains an exact supporting "
          "span found in its fragment; unverifiable spans were dropped.", ""]

    L += ["## 4. Top question prototypes", ""]
    L += _md_table(["Prototype", "Intent", "Fragments", "Docs", "Density"],
                   [[p["canonical_name"], p["intent"],
                     p["support_fragment_count"], p["support_document_count"],
                     density.get(p["prototype_id"], 0.0)]
                    for p in prototypes[:15]])
    L.append("")

    L += ["## 5. Qspace tracker summary", "",
          f"- Node counts: {ctx['tracker']['counts']}",
          f"- Edge counts: {ctx['tracker']['edge_counts']}",
          f"- Thin/coverage-gap prototypes: "
          f"{len(ctx['tracker']['missing_evidence_regions'])}", ""]

    L += ["## 6. Proposed classes / question subspaces", ""]
    for c in classes:
        L += [f"### {c['class_id']} — {c['name']} (status: {c['status']})",
              f"- Dominant intents: {', '.join(c['dominant_intents'])}",
              f"- Common evidence roles: {', '.join(c['common_evidence_roles'])}",
              f"- Class objectives: {', '.join(c['class_objectives'])}",
              f"- Members: {c['size_prototypes']} prototypes, "
              f"{c['size_fragments']} fragments across "
              f"{len(c['member_documents'])} documents", ""]

    L += ["## 7. Question density analysis", "",
          "Density blends supporting-fragment count, supporting-document "
          "count, an intent risk prior, and synthetic frequency (no real query "
          "logs are available). Top prototypes by density:", ""]
    dens_sorted = sorted(prototypes,
                         key=lambda p: -density.get(p["prototype_id"], 0.0))[:8]
    L += _md_table(["Prototype", "Density"],
                   [[p["canonical_name"], density.get(p["prototype_id"], 0.0)]
                    for p in dens_sorted])
    L.append("")

    L += ["## 8. Question proximity analysis", "",
          "Proximity is an inspectable weighted blend of intent, evidence-role "
          "Jaccard, slot Jaccard, answer-shape Jaccard, fragment overlap, and "
          "class overlap. Closest prototype pairs:", ""]
    L += _md_table(["A", "B", "Similarity"],
                   [[r["name_a"], r["name_b"], r["similarity"]]
                    for r in ctx["proximity_rows"][:8]])
    L.append("")

    L += ["## 9. Evidence role analysis", "",
          f"- Evidence units extracted: **{ctx['num_evidence']}**",
          f"- Distinct roles: {ctx['num_evidence_roles']}",
          "- Strong co-demand pairs are visualized in "
          "`plots/evidence_codemand_heatmap.png`.", ""]

    L += ["## 10. Retrieval tree compilation results", "",
          f"- Trees compiled: **{len(trees)}**",
          f"- Selected type distribution: {ctx['selected_tree_types']}",
          "- Each fragment is compiled into 5 candidate trees "
          "(source_order, evidence_role, question_intent, class_objective, "
          "hybrid_optimized); the tree minimizing J(T) is selected.", ""]

    L += ["## 11. Candidate tree comparison (mean total_cost)", ""]
    L += _md_table(["Candidate type", "Mean total_cost"],
                   [[k, round(v, 4)] for k, v in ctx["mean_cost_by_type"].items()])
    L.append("")

    L += ["## 12. Sample optimized trees", ""]
    for tr in trees[:3]:
        sc = tr["score"]
        L += [f"### {tr['tree_id']} ({tr['selected_candidate_type']})",
              f"- optimized_for: {', '.join(tr['optimized_for'])}",
              f"- expected_path_length={sc['expected_path_length']}, "
              f"expected_context_tokens={sc['expected_context_tokens']}, "
              f"trace_clarity={sc['trace_clarity']}, "
              f"total_cost={sc['total_cost']}, J={sc['objective_J']}",
              f"- visual: `trees/tree_{tr['fragment_id']}.png` / "
              f"`.html` / `.dot`", ""]

    L += ["## 13. Sample retrieval traces", "",
          "See `reports/sample_retrieval_traces.md` for full traces. "
          f"{ctx['num_sims']} query traversals were simulated.", ""]

    L += ["## 14. Baseline comparison", ""]
    L += _md_table(["Strategy", "Evid. recall", "Avg tokens", "Avg path",
                    "Avg cost"],
                   [[r["strategy"], r["evidence_recall"],
                     r["avg_context_tokens"], r["avg_path_length"],
                     r["avg_total_cost"]] for r in eval_summary])
    L.append("")
    if improvements:
        L += ["**Improvements:**", ""]
        for k, v in improvements.items():
            L.append(f"- {k}: {v}%")
        L.append("")

    L += ["## 15. Known limitations", "",
          "- **Heuristic detection.** Answer affordances come from transparent "
          "regex/rule detectors, not an LLM. Recall is bounded by the pattern "
          "set; novel phrasings may be missed.",
          "- **Span validation** guarantees each direct question quotes real "
          "fragment text, but does not guarantee the question is well-posed.",
          "- **Generated questions may be incomplete**; partial affordances "
          "flag missing evidence but are not exhaustively enumerated.",
          "- **Class discovery is provisional** (status = proposed) and "
          "produced by clustering an explicit intent/evidence feature space.",
          "- **Density is synthetic**, derived from support counts and an "
          "intent risk prior; there are no real query logs.",
          "- **Proximity is heuristic**, a fixed weighted similarity.",
          "- **Tree optimization is heuristic**: five candidate structures are "
          "scored and the best J(T) is chosen; it is not a global optimum over "
          "all possible trees.",
          "- **Baseline costs are modeled** with transparent assumptions, not "
          "measured against a live LLM retriever.", ""]

    L += ["## 16. Next recommended improvements", "",
          "- Plug in an LLM for affordance detection and prototype naming, "
          "keeping the span-validation gate.",
          "- Learn density from real query logs / click data.",
          "- Replace the fixed proximity weights with a learned metric.",
          "- Add a global tree search (e.g. cost-guided beam search) beyond the "
          "five templates.",
          "- Add embeddings as an auxiliary clustering/visualization aid.",
          "- Expand evaluation to a live retrieval harness with answer grading.",
          ""]

    (paths.reports / "final_report.md").write_text("\n".join(L) + "\n",
                                                   encoding="utf-8")


def _run_summary(paths: Paths, ctx: Dict) -> None:
    eval_summary = ctx["eval"]["summary"]
    qs = next((r for r in eval_summary
               if r["strategy"] == "qspace_optimized_tree"), {})
    L = ["# Qspace Pipeline — Run Summary", "",
         f"_Generated: {datetime.now().isoformat(timespec='seconds')}_", "",
         f"- Documents: {ctx['corpus_meta']['total_documents']}",
         f"- Fragments: {ctx['num_fragments']}",
         f"- Direct questions: {ctx['num_direct']} | Partial: {ctx['num_partial']}",
         f"- Question prototypes: {len(ctx['prototypes'])}",
         f"- Proposed classes: {len(ctx['classes'])}",
         f"- Evidence units: {ctx['num_evidence']}",
         f"- Retrieval trees compiled: {len(ctx['trees'])}",
         f"- Query traversals simulated: {ctx['num_sims']}",
         f"- Detection method: {'LLM' if ctx['used_llm'] else 'heuristic fallback'}",
         "",
         "## Headline evaluation (qspace_optimized_tree)", "",
         f"- evidence_recall: {qs.get('evidence_recall', 'n/a')}",
         f"- avg_context_tokens: {qs.get('avg_context_tokens', 'n/a')}",
         f"- avg_path_length: {qs.get('avg_path_length', 'n/a')}",
         f"- avg_total_cost: {qs.get('avg_total_cost', 'n/a')}", ""]
    if ctx["eval"]["improvements"]:
        L += ["## Improvements vs baselines", ""]
        for k, v in ctx["eval"]["improvements"].items():
            L.append(f"- {k}: {v}%")
    (paths.reports / "run_summary.md").write_text("\n".join(L) + "\n",
                                                  encoding="utf-8")


def _dashboard(paths: Paths, ctx: Dict) -> None:
    plots_dir = paths.plots
    plot_files = sorted(p.name for p in plots_dir.glob("*.png")) \
        if plots_dir.exists() else []
    tree_pngs = sorted(p.name for p in paths.trees.glob("*.png")) \
        if paths.trees.exists() else []

    def esc(s):
        return html.escape(str(s))

    cards = "".join(
        f"<figure><img src='plots/{esc(f)}' loading='lazy'>"
        f"<figcaption>{esc(f)}</figcaption></figure>" for f in plot_files)
    tree_cards = "".join(
        f"<figure><a href='trees/{esc(f)}'><img src='trees/{esc(f)}' "
        f"loading='lazy'></a><figcaption>{esc(f)}</figcaption></figure>"
        for f in tree_pngs)

    eval_rows = "".join(
        "<tr>" + "".join(f"<td>{esc(v)}</td>" for v in
                         (r["strategy"], r["evidence_recall"],
                          r["avg_context_tokens"], r["avg_path_length"],
                          r["avg_total_cost"])) + "</tr>"
        for r in ctx["eval"]["summary"])

    artifacts = [
        "reports/final_report.md", "reports/run_summary.md",
        "reports/sample_retrieval_traces.md", "corpus_inventory.csv",
        "question_prototypes.csv", "classes.csv", "question_density.csv",
        "question_proximity.csv", "evidence_roles.csv",
        "retrieval_trees.jsonl", "candidate_tree_scores.csv",
        "selected_tree_scores.csv", "retrieval_simulations.csv",
        "evaluation/baseline_comparison.csv", "qspace_tracker.json"]
    art_links = "".join(f"<li><a href='{a}'>{a}</a></li>" for a in artifacts)

    html_doc = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Qspace Knowledge Compilation Dashboard</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#0f1220;color:#e8eaf2}}
header{{padding:24px 32px;background:#1b2140}}
h1{{margin:0;font-size:22px}} h2{{border-bottom:1px solid #333a5c;padding-bottom:6px}}
main{{padding:24px 32px;max-width:1400px;margin:auto}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}}
figure{{margin:0;background:#171b30;border:1px solid #2a3050;border-radius:8px;padding:8px}}
figure img{{width:100%;border-radius:4px;background:#fff}}
figcaption{{font-size:12px;color:#9aa3c7;margin-top:6px;word-break:break-all}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}
td,th{{border:1px solid #2a3050;padding:6px 10px;font-size:14px}}
th{{background:#1b2140;text-align:left}}
a{{color:#8fb3ff}} ul{{columns:2}} .stat{{display:inline-block;background:#1b2140;
padding:10px 16px;border-radius:8px;margin:4px}}
</style></head><body>
<header><h1>Question-Space Driven Knowledge Compilation</h1>
<div>Generated {esc(datetime.now().isoformat(timespec='seconds'))} ·
detection: {'LLM' if ctx['used_llm'] else 'heuristic fallback'}</div></header>
<main>
<section><h2>Overview</h2>
<span class='stat'>Documents: {ctx['corpus_meta']['total_documents']}</span>
<span class='stat'>Fragments: {ctx['num_fragments']}</span>
<span class='stat'>Direct Qs: {ctx['num_direct']}</span>
<span class='stat'>Prototypes: {len(ctx['prototypes'])}</span>
<span class='stat'>Classes: {len(ctx['classes'])}</span>
<span class='stat'>Evidence: {ctx['num_evidence']}</span>
<span class='stat'>Trees: {len(ctx['trees'])}</span>
<span class='stat'>Simulations: {ctx['num_sims']}</span></section>
<section><h2>Baseline comparison</h2><table>
<tr><th>Strategy</th><th>Evidence recall</th><th>Avg tokens</th>
<th>Avg path</th><th>Avg cost</th></tr>{eval_rows}</table></section>
<section><h2>Key artifacts</h2><ul>{art_links}</ul></section>
<section><h2>Optimized retrieval trees</h2><div class='grid'>{tree_cards}</div></section>
<section><h2>All plots</h2><div class='grid'>{cards}</div></section>
</main></body></html>"""
    (paths.out / "dashboard.html").write_text(html_doc, encoding="utf-8")
