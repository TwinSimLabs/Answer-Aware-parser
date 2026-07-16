"""Components 8 & 9: question density and proximity estimation."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from . import plotting
from .config import (DEFAULT_RISK_WEIGHT, INTENT_RISK_WEIGHT,
                     PROXIMITY_WEIGHTS, Paths)
from .utils import banner, jaccard, log, sub, write_csv, write_json


def estimate_density(paths: Paths, prototypes: List[Dict],
                     proto_to_class: Dict[str, str],
                     classes: List[Dict]) -> Dict[str, float]:
    banner("DENSITY", "Question density estimation (synthetic)")
    if not prototypes:
        return {}
    max_frag = max(p["support_fragment_count"] for p in prototypes) or 1
    max_doc = max(p["support_document_count"] for p in prototypes) or 1
    total_frag = sum(p["support_fragment_count"] for p in prototypes) or 1

    rows: List[Dict] = []
    density: Dict[str, float] = {}
    for p in prototypes:
        synth_freq = p["support_fragment_count"] / total_frag
        risk = INTENT_RISK_WEIGHT.get(p["intent"], DEFAULT_RISK_WEIGHT)
        support_norm = p["support_fragment_count"] / max_frag
        doc_norm = p["support_document_count"] / max_doc
        importance = 0.5 * risk + 0.3 * support_norm + 0.2 * doc_norm
        final = round(0.35 * synth_freq * (max_frag / 1.0) / max(1, len(prototypes))
                      + 0.35 * support_norm + 0.30 * importance, 4)
        # keep final in a readable 0..1 range
        final = round(min(1.0, 0.45 * support_norm + 0.35 * importance
                          + 0.20 * doc_norm), 4)
        density[p["prototype_id"]] = final
        rows.append({
            "prototype_id": p["prototype_id"],
            "canonical_name": p["canonical_name"],
            "class_id": proto_to_class.get(p["prototype_id"], ""),
            "support_fragment_count": p["support_fragment_count"],
            "support_document_count": p["support_document_count"],
            "synthetic_frequency": round(synth_freq, 4),
            "observed_frequency": 0.0,
            "inferred_importance": round(importance, 4),
            "risk_weight": round(risk, 3),
            "final_density": final,
        })

    rows.sort(key=lambda r: -r["final_density"])
    write_csv(paths.out / "question_density.csv", rows)

    # class density = mean of member prototype densities
    class_rows: List[Dict] = []
    for c in classes:
        vals = [density.get(pid, 0.0) for pid in c["question_prototypes"]]
        cd = round(sum(vals) / len(vals), 4) if vals else 0.0
        class_rows.append({"class_id": c["class_id"], "class_name": c["name"],
                           "class_density": cd,
                           "member_prototypes": len(vals)})
    write_csv(paths.out / "class_density.csv", class_rows)

    log("DENSITY", "Top dense question prototypes:")
    for r in rows[:6]:
        sub(f"{r['canonical_name']} density={r['final_density']}")

    plotting.hist(paths.plots / "question_density_distribution.png",
                  [r["final_density"] for r in rows],
                  "Question density distribution", "final_density", bins=12)
    plotting.bar(paths.plots / "top_question_density.png",
                 [r["canonical_name"] for r in rows[:12]],
                 [r["final_density"] for r in rows[:12]],
                 "Top question density", ylabel="density", horizontal=True)
    plotting.bar(paths.plots / "class_density_radar_or_bar.png",
                 [r["class_name"] for r in class_rows],
                 [r["class_density"] for r in class_rows],
                 "Class density", ylabel="mean density", color="#8172B3")
    log("DENSITY", f"Saved: {paths.out / 'question_density.csv'}")
    return density


def _similarity(a: Dict, b: Dict, class_of: Dict[str, str]) -> Dict[str, float]:
    intent_sim = 1.0 if a["intent"] == b["intent"] else 0.0
    role_j = jaccard(a["required_evidence_roles"], b["required_evidence_roles"])
    slot_j = jaccard(a["slots"], b["slots"])
    shape_j = jaccard(a["answer_shape"], b["answer_shape"])
    frag_o = jaccard(a["support_fragment_ids"], b["support_fragment_ids"])
    class_o = 1.0 if (class_of.get(a["prototype_id"])
                      == class_of.get(b["prototype_id"])
                      and class_of.get(a["prototype_id"])) else 0.0
    parts = {
        "intent": intent_sim, "evidence_role_jaccard": role_j,
        "slot_jaccard": slot_j, "answer_shape_jaccard": shape_j,
        "fragment_overlap": frag_o, "class_overlap": class_o,
    }
    total = sum(PROXIMITY_WEIGHTS[k] * v for k, v in parts.items())
    parts["total"] = round(total, 4)
    return parts


def estimate_proximity(paths: Paths, prototypes: List[Dict],
                       proto_to_class: Dict[str, str]
                       ) -> Tuple[List[Dict], List[List[float]], List[str]]:
    banner("PROXIMITY", "Question proximity estimation")
    ids = [p["prototype_id"] for p in prototypes]
    names = [p["canonical_name"] for p in prototypes]
    n = len(prototypes)
    matrix = [[0.0] * n for _ in range(n)]
    rows: List[Dict] = []
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            parts = _similarity(prototypes[i], prototypes[j], proto_to_class)
            matrix[i][j] = matrix[j][i] = parts["total"]
            rows.append({
                "prototype_a": ids[i], "name_a": names[i],
                "prototype_b": ids[j], "name_b": names[j],
                "intent_similarity": round(parts["intent"], 3),
                "evidence_role_jaccard": round(parts["evidence_role_jaccard"], 3),
                "slot_jaccard": round(parts["slot_jaccard"], 3),
                "answer_shape_jaccard": round(parts["answer_shape_jaccard"], 3),
                "fragment_overlap": round(parts["fragment_overlap"], 3),
                "class_overlap": round(parts["class_overlap"], 3),
                "similarity": parts["total"],
            })
    rows.sort(key=lambda r: -r["similarity"])
    write_csv(paths.out / "question_proximity.csv", rows)
    write_json(paths.out / "question_proximity_matrix.json",
               {"ids": ids, "names": names, "matrix": matrix})

    log("PROXIMITY", "Nearest neighbors (top similar pairs):")
    for r in rows[:8]:
        sub(f"{r['name_a']}")
        sub(f"    -> {r['name_b']} score={r['similarity']}", 4)

    if n:
        plotting.heatmap(paths.plots / "question_proximity_heatmap.png",
                         matrix, names, names,
                         "Question proximity", cmap="viridis",
                         annotate=n <= 16)
    _proximity_network(paths, prototypes, matrix, proto_to_class)
    log("PROXIMITY", f"Saved: {paths.out / 'question_proximity.csv'}")
    return rows, matrix, ids


def _proximity_network(paths: Paths, prototypes, matrix, proto_to_class) -> None:
    if not plotting.available():
        return
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception:
        return
    G = nx.Graph()
    for p in prototypes:
        G.add_node(p["prototype_id"], label=p["intent"])
    n = len(prototypes)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] >= 0.45:
                G.add_edge(prototypes[i]["prototype_id"],
                           prototypes[j]["prototype_id"],
                           weight=matrix[i][j])
    if G.number_of_nodes() == 0:
        return
    classes = sorted(set(proto_to_class.values()))
    cmap = {c: i for i, c in enumerate(classes)}
    colors = [cmap.get(proto_to_class.get(nd, ""), -1) for nd in G.nodes()]
    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(G, seed=42, k=0.6)
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap="tab10",
                           node_size=500, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax,
                           width=[G[u][v]["weight"] * 2 for u, v in G.edges()])
    labels = {nd: G.nodes[nd]["label"] for nd in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    ax.set_title("Question proximity network (edges >= 0.45)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(paths.plots / "question_proximity_network.png", dpi=110)
    plt.close(fig)
