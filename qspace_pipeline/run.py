"""End-to-end pipeline orchestrator."""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Optional

from . import (affordances, classes, corpus_discovery, evaluation, evidence,
               fragmentation, loading, planner, prototypes, proximity,
               qspace_tracker, reporting, trees)
from .config import Paths
from .utils import banner, log, sub, write_json


def run(project_root: Optional[Path] = None) -> None:
    t0 = time.time()
    root = Path(project_root or Path.cwd()).resolve()
    paths = Paths(project_root=root)
    paths.ensure()

    banner("START", "Question-Space Driven Knowledge Compilation")
    log("START", f"Run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("START", f"Project root: {root}")
    log("START", f"Outputs: {paths.out}")

    # 1-2 discovery + inventory
    inventory, corpus_meta = corpus_discovery.discover_and_inventory(paths)
    # 2 loading
    documents = loading.load_documents(paths, inventory)
    # refresh inventory with load status
    from .utils import write_csv as _wc
    _wc(paths.out / "corpus_inventory.csv", inventory)

    # 3 fragmentation
    fragments = fragmentation.fragment_documents(paths, documents)
    # 4 affordances
    affs = affordances.detect_affordances(paths, fragments)
    # 5 prototypes
    protos, edges = prototypes.normalize_prototypes(paths, affs)
    # 7 classes (needs prototypes + affordances)
    class_list, proto_to_class = classes.discover_classes(paths, protos, affs)
    # 8 density
    density = proximity.estimate_density(paths, protos, proto_to_class,
                                         class_list)
    # 9 proximity
    prox_rows, prox_matrix, prox_ids = proximity.estimate_proximity(
        paths, protos, proto_to_class)
    # 10 evidence roles
    ev = evidence.extract_evidence(paths, affs, edges, class_list)
    # 11 co-demand (needs density)
    codemand_rows = evidence.compute_codemand(paths, protos, density)
    # 6 qspace tracker
    tracker = qspace_tracker.build_tracker(paths, protos, edges, ev, class_list,
                                           density, prox_rows, proto_to_class)
    # 12 trees
    tree_list, selected_rows = trees.compile_trees(
        paths, fragments, edges, ev, protos, density, class_list,
        proto_to_class, codemand_rows)
    # 13 planner
    sims = planner.simulate_queries(paths, protos, edges, tree_list, ev,
                                    proto_to_class, affs)
    # 14 evaluation
    eval_result = evaluation.evaluate(paths, affs, edges, protos, tree_list, ev,
                                      fragments)

    # aggregates for reporting
    num_direct = sum(len(r["directly_answerable"]) for r in affs)
    num_partial = sum(len(r["partially_answerable"]) for r in affs)
    frag_type_counts = dict(Counter(f["fragment_type"] for f in fragments))
    selected_tree_types = dict(Counter(t["selected_candidate_type"]
                                       for t in tree_list))
    # mean candidate cost by type
    from collections import defaultdict
    import csv as _csv
    cost_by_type = defaultdict(list)
    cand_csv = paths.out / "candidate_tree_scores.csv"
    if cand_csv.exists():
        with cand_csv.open(encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                try:
                    cost_by_type[row["candidate_type"]].append(
                        float(row["total_cost"]))
                except (ValueError, KeyError):
                    pass
    mean_cost_by_type = {k: sum(v) / len(v) for k, v in cost_by_type.items()}

    banner("REPORT", "Report + dashboard generation")
    ctx = {
        "corpus_meta": corpus_meta,
        "prototypes": protos,
        "classes": class_list,
        "density": density,
        "proximity_rows": prox_rows,
        "trees": tree_list,
        "tracker": tracker,
        "eval": eval_result,
        "used_llm": affordances.USED_LLM,
        "num_fragments": len(fragments),
        "fragment_type_counts": frag_type_counts,
        "num_affordance_frags": len(affs),
        "num_direct": num_direct,
        "num_partial": num_partial,
        "num_evidence": len(ev),
        "num_evidence_roles": len({e["role"] for e in ev}),
        "num_sims": len(sims),
        "selected_tree_types": selected_tree_types,
        "mean_cost_by_type": mean_cost_by_type,
    }
    reporting.write_reports(paths, ctx)
    write_json(paths.out / "run_metadata.json", {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "project_root": str(root),
        "elapsed_seconds": round(time.time() - t0, 2),
        "counts": {
            "documents": corpus_meta["total_documents"],
            "fragments": len(fragments),
            "direct_questions": num_direct,
            "partial_questions": num_partial,
            "prototypes": len(protos),
            "classes": len(class_list),
            "evidence_units": len(ev),
            "trees": len(tree_list),
            "simulations": len(sims),
        },
        "used_llm": affordances.USED_LLM,
    })
    log("REPORT", f"final_report.md, run_summary.md, dashboard.html written to "
                  f"{paths.reports} / {paths.out}")

    _final_banner(paths, time.time() - t0)


def _final_banner(paths: Paths, elapsed: float) -> None:
    banner("DONE", "Qspace pipeline completed")
    print(f"[DONE] Elapsed: {elapsed:.1f}s\n")
    print("Main reports:")
    for p in ("reports/run_summary.md", "reports/final_report.md",
              "dashboard.html"):
        sub(f"- outputs/qspace/{p}")
    print("\nKey data:")
    for p in ("question_prototypes.csv", "classes.csv",
              "retrieval_trees.jsonl", "evaluation/baseline_comparison.csv"):
        sub(f"- outputs/qspace/{p}")
    print("\nKey plots:")
    for p in ("question_prototype_frequency.png", "class_intent_heatmap.png",
              "question_proximity_heatmap.png",
              "candidate_tree_cost_comparison.png",
              "baseline_vs_qspace_context_tokens.png"):
        sub(f"- outputs/qspace/plots/{p}")
    print()
