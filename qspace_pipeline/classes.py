"""Component 7: class / question-subspace discovery.

Classes emerge from clusters of question prototypes that share intents,
evidence roles, answer shapes, and overlapping supporting fragments -- NOT from
document file names. Clustering runs on an explicit feature space (intent +
evidence-role + answer-shape indicators). Classes are marked ``proposed``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from . import plotting
from .config import Paths
from .utils import banner, jaccard, log, sub, write_csv, write_json

# Reference archetypes used only to *name* discovered clusters (not to force
# membership). Naming is by dominant-intent overlap.
_ARCHETYPES = {
    "Policy-like Fragments": {
        "applicability_check", "rule_lookup", "exception_lookup",
        "approval_lookup", "owner_lookup", "effective_date_lookup",
        "threshold_lookup"},
    "Finance-like Fragments": {
        "metric_lookup", "variance_explanation", "threshold_lookup"},
    "SOP-like Fragments": {
        "next_step_lookup", "sla_lookup", "escalation_lookup",
        "owner_lookup", "input_output_lookup"},
}

_CLASS_OBJECTIVES = {
    "Policy-like Fragments": ["applicability_check", "rule_lookup",
                              "exception_lookup", "approval_lookup",
                              "owner_lookup", "effective_date_lookup"],
    "Finance-like Fragments": ["metric_lookup", "variance_explanation",
                               "threshold_lookup", "source_table_trace",
                               "forecast_lookup"],
    "SOP-like Fragments": ["next_step_lookup", "owner_per_step_lookup",
                           "sla_lookup", "escalation_lookup",
                           "input_output_lookup"],
}


def _feature_vector(proto: Dict, all_intents: List[str],
                    all_roles: List[str]) -> List[float]:
    vec = [0.0] * (len(all_intents) + len(all_roles))
    if proto["intent"] in all_intents:
        vec[all_intents.index(proto["intent"])] = 1.0
    for r in proto["required_evidence_roles"]:
        if r in all_roles:
            vec[len(all_intents) + all_roles.index(r)] = 1.0
    return vec


def _name_cluster(intents: Counter) -> Tuple[str, float]:
    total = sum(intents.values()) or 1
    top_intent, top_n = intents.most_common(1)[0]
    # A cluster dominated by a single intent is named as that intent's subspace.
    if top_n / total >= 0.6 and len(intents) <= 2:
        pretty = top_intent.replace("_", " ").title()
        return f"{pretty} Subspace", round(top_n / total, 3)
    best, best_score = "Mixed Fragments", 0.0
    dom = set(i for i, _ in intents.most_common(4))
    for name, arche in _ARCHETYPES.items():
        score = jaccard(dom, arche)
        if score > best_score:
            best, best_score = name, score
    return best, round(best_score, 3)


def discover_classes(paths: Paths, prototypes: List[Dict],
                     affordances: List[Dict]) -> Tuple[List[Dict], Dict[str, str]]:
    banner("CLASS", "Class / question-subspace discovery")
    if not prototypes:
        log("CLASS", "No prototypes; skipping class discovery.")
        return [], {}

    all_intents = sorted({p["intent"] for p in prototypes})
    all_roles = sorted({r for p in prototypes
                        for r in p["required_evidence_roles"]})
    X = [_feature_vector(p, all_intents, all_roles) for p in prototypes]

    n_clusters = max(2, min(5, len(prototypes) // 2))
    labels: List[int]
    method = "agglomerative"
    try:
        from sklearn.cluster import AgglomerativeClustering
        import numpy as np
        arr = np.array(X)
        if len(prototypes) <= n_clusters:
            labels = list(range(len(prototypes)))
        else:
            labels = list(AgglomerativeClustering(
                n_clusters=n_clusters).fit_predict(arr))
    except Exception as exc:  # deterministic fallback: group by dominant intent archetype
        method = "intent-archetype-fallback"
        log("CLASS", f"sklearn unavailable ({exc}); using archetype fallback.")
        labels = []
        for p in prototypes:
            assigned = 0
            for i, (name, arche) in enumerate(_ARCHETYPES.items()):
                if p["intent"] in arche:
                    assigned = i
                    break
            labels.append(assigned)

    clusters: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        clusters[lab].append(i)

    # map fragment -> documents for member documents
    frag_doc: Dict[str, str] = {}
    for rec in affordances:
        frag_doc[rec["fragment_id"]] = rec["document_id"]

    classes: List[Dict] = []
    proto_to_class: Dict[str, str] = {}
    frag_class_rows: List[Dict] = []
    q_class_rows: List[Dict] = []
    used_names: Counter = Counter()

    for lab, idxs in sorted(clusters.items()):
        members = [prototypes[i] for i in idxs]
        intents = Counter(m["intent"] for m in members)
        roles = Counter(r for m in members
                        for r in m["required_evidence_roles"])
        frag_ids = sorted({f for m in members for f in m["support_fragment_ids"]})
        doc_ids = sorted({frag_doc.get(f, "?") for f in frag_ids} - {"?"})
        name, score = _name_cluster(intents)
        used_names[name] += 1
        suffix = f" #{used_names[name]}" if used_names[name] > 1 else ""
        display_name = name + suffix
        cid = f"class_{lab + 1:03d}"
        objectives = _CLASS_OBJECTIVES.get(name,
                                           [i for i, _ in intents.most_common(5)])
        cls = {
            "class_id": cid,
            "name": display_name,
            "status": "proposed",
            "naming_confidence": score,
            "description": (f"Proposed question-subspace dominated by "
                            f"{', '.join(i for i, _ in intents.most_common(3))}."),
            "question_prototypes": [m["prototype_id"] for m in members],
            "dominant_intents": [i for i, _ in intents.most_common(5)],
            "common_evidence_roles": [r for r, _ in roles.most_common(6)],
            "member_fragments": frag_ids,
            "member_documents": doc_ids,
            "class_objectives": objectives,
            "size_prototypes": len(members),
            "size_fragments": len(frag_ids),
        }
        classes.append(cls)
        for m in members:
            proto_to_class[m["prototype_id"]] = cid
            m["class_candidates"] = [cid]
            q_class_rows.append({"prototype_id": m["prototype_id"],
                                 "class_id": cid, "class_name": display_name})
        for f in frag_ids:
            frag_class_rows.append({"fragment_id": f, "class_id": cid,
                                    "class_name": display_name})

    write_json(paths.out / "classes.json", classes)
    write_csv(paths.out / "classes.csv", classes, columns=[
        "class_id", "name", "status", "naming_confidence", "size_prototypes",
        "size_fragments", "dominant_intents", "common_evidence_roles",
        "class_objectives", "member_documents"])
    write_csv(paths.out / "fragment_class_membership.csv", frag_class_rows,
              columns=["fragment_id", "class_id", "class_name"])
    write_csv(paths.out / "question_class_membership.csv", q_class_rows,
              columns=["prototype_id", "class_id", "class_name"])

    log("CLASS", f"Clustering method: {method}")
    log("CLASS", f"Proposed classes: {len(classes)}")
    for c in classes:
        sub(f"{c['class_id']}: {c['name']} (status={c['status']})")
        sub(f"    prototypes: {c['size_prototypes']} | fragments: {c['size_fragments']}", 4)
        sub(f"    dominant intents: {', '.join(c['dominant_intents'][:4])}", 4)
        sub(f"    evidence roles: {', '.join(c['common_evidence_roles'][:5])}", 4)

    _class_plots(paths, classes, all_intents, all_roles, frag_doc)
    log("CLASS", f"Saved: {paths.out / 'classes.json'}")
    return classes, proto_to_class


def _class_plots(paths: Paths, classes: List[Dict], all_intents, all_roles,
                 frag_doc) -> None:
    if not classes:
        return
    names = [c["name"] for c in classes]
    plotting.bar(paths.plots / "class_size_distribution.png",
                 names, [c["size_fragments"] for c in classes],
                 "Class size (fragments)", ylabel="fragments", horizontal=True)

    # intent heatmap: class x intent
    intent_mat = []
    for c in classes:
        counts = Counter(c["dominant_intents"])
        # weight by presence
        row = [1.0 if it in c["dominant_intents"] else 0.0 for it in all_intents]
        intent_mat.append(row)
    plotting.heatmap(paths.plots / "class_intent_heatmap.png",
                     intent_mat, names, all_intents,
                     "Class x dominant intent", cmap="Blues", annotate=True)

    role_mat = [[1.0 if r in c["common_evidence_roles"] else 0.0
                 for r in all_roles] for c in classes]
    plotting.heatmap(paths.plots / "class_evidence_role_heatmap.png",
                     role_mat, names, all_roles,
                     "Class x evidence role", cmap="Greens", annotate=True)

    # document-class overlap: doc x class membership counts
    docs = sorted({d for c in classes for d in c["member_documents"]})
    if docs:
        overlap = [[1.0 if d in c["member_documents"] else 0.0
                    for c in classes] for d in docs]
        plotting.heatmap(paths.plots / "document_class_overlap.png",
                         overlap, docs, names,
                         "Document x class overlap", cmap="Purples",
                         annotate=len(docs) * len(classes) <= 200)
