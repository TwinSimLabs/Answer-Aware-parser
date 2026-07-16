"""Component 6: persistent Qspace tracker and graph."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

from . import plotting
from .config import Paths
from .utils import banner, log, sub, write_csv, write_json


def build_tracker(paths: Paths, prototypes: List[Dict], edges: List[Dict],
                  evidence: List[Dict], classes: List[Dict],
                  density: Dict[str, float], proximity_rows: List[Dict],
                  proto_to_class: Dict[str, str]) -> Dict:
    banner("QSPACE", "Qspace tracker + graph")

    nodes: List[Dict] = []
    graph_edges: List[Dict] = []

    proto_ids = {p["prototype_id"] for p in prototypes}
    frag_ids = {e["fragment_id"] for e in edges}
    role_ids = {ev["role"] for ev in evidence}
    doc_ids = {ev["document_id"] for ev in evidence}

    for p in prototypes:
        nodes.append({"node_id": p["prototype_id"], "node_type": "prototype",
                      "label": p["canonical_name"],
                      "density": density.get(p["prototype_id"], 0.0)})
    for f in sorted(frag_ids):
        nodes.append({"node_id": f, "node_type": "fragment", "label": f,
                      "density": ""})
    for r in sorted(role_ids):
        nodes.append({"node_id": f"role::{r}", "node_type": "evidence_role",
                      "label": r, "density": ""})
    for c in classes:
        nodes.append({"node_id": c["class_id"], "node_type": "class",
                      "label": c["name"], "density": ""})
    for d in sorted(doc_ids):
        nodes.append({"node_id": d, "node_type": "document", "label": d,
                      "density": ""})

    # prototype-fragment edges
    for e in edges:
        if e["prototype_id"] in proto_ids:
            graph_edges.append({"source": e["prototype_id"],
                                "target": e["fragment_id"],
                                "edge_type": "prototype_fragment",
                                "weight": e["confidence"]})
    # prototype-evidence_role edges
    for p in prototypes:
        for r in p["required_evidence_roles"]:
            graph_edges.append({"source": p["prototype_id"],
                                "target": f"role::{r}",
                                "edge_type": "prototype_evidence",
                                "weight": 1.0})
    # prototype-class
    for p in prototypes:
        cid = proto_to_class.get(p["prototype_id"])
        if cid:
            graph_edges.append({"source": p["prototype_id"], "target": cid,
                                "edge_type": "prototype_class", "weight": 1.0})
    # fragment-document and fragment-evidence_role
    frag_doc: Dict[str, str] = {}
    for ev in evidence:
        frag_doc[ev["fragment_id"]] = ev["document_id"]
        graph_edges.append({"source": ev["fragment_id"],
                            "target": f"role::{ev['role']}",
                            "edge_type": "fragment_evidence", "weight": 1.0})
    for f, d in frag_doc.items():
        graph_edges.append({"source": f, "target": d,
                            "edge_type": "fragment_document", "weight": 1.0})

    write_csv(paths.out / "qspace_graph_nodes.csv", nodes,
              columns=["node_id", "node_type", "label", "density"])
    write_csv(paths.out / "qspace_graph_edges.csv", graph_edges,
              columns=["source", "target", "edge_type", "weight"])

    # degree by prototype
    deg: Counter[str] = Counter()
    for e in graph_edges:
        if e["edge_type"] == "prototype_fragment":
            deg[e["source"]] += 1
    name_by_id = {p["prototype_id"]: p["canonical_name"] for p in prototypes}

    edge_type_counts = Counter(e["edge_type"] for e in graph_edges)
    tracker = {
        "counts": {
            "prototypes": len(proto_ids),
            "fragments": len(frag_ids),
            "evidence_roles": len(role_ids),
            "classes": len(classes),
            "documents": len(doc_ids),
        },
        "edge_counts": dict(edge_type_counts),
        "prototypes": [{
            "prototype_id": p["prototype_id"],
            "canonical_name": p["canonical_name"],
            "intent": p["intent"],
            "density": density.get(p["prototype_id"], 0.0),
            "class": proto_to_class.get(p["prototype_id"], ""),
            "support_fragments": p["support_fragment_count"],
            "support_documents": p["support_document_count"],
            "example_raw_questions": p["example_raw_questions"],
        } for p in prototypes],
        "highest_degree_prototypes": [
            {"prototype_id": pid, "name": name_by_id.get(pid, pid),
             "fragment_degree": d} for pid, d in deg.most_common(10)],
        "missing_evidence_regions": _missing_regions(prototypes),
    }
    write_json(paths.out / "qspace_tracker.json", tracker)

    log("QSPACE", "Nodes:")
    sub(f"prototypes: {len(proto_ids)}")
    sub(f"fragments: {len(frag_ids)}")
    sub(f"evidence roles: {len(role_ids)}")
    sub(f"classes: {len(classes)} | documents: {len(doc_ids)}")
    log("QSPACE", "Edges:")
    for et, c in edge_type_counts.most_common():
        sub(f"{et}: {c}")
    log("QSPACE", "Highest-degree prototypes:")
    for pid, d in deg.most_common(5):
        sub(f"{name_by_id.get(pid, pid)}: {d} fragments")

    _bipartite_plot(paths, prototypes, edges, deg, name_by_id)
    _intent_graph_plot(paths, prototypes, proximity_rows)
    log("QSPACE", f"Saved: {paths.out / 'qspace_tracker.json'}")
    return tracker


def _missing_regions(prototypes: List[Dict]) -> List[Dict]:
    """Prototypes with low support flagged as thin/missing-evidence regions."""
    out = []
    for p in prototypes:
        if p["support_fragment_count"] <= 1:
            out.append({"prototype_id": p["prototype_id"],
                        "canonical_name": p["canonical_name"],
                        "support_fragments": p["support_fragment_count"],
                        "note": "thin support; candidate coverage gap"})
    return out


def _bipartite_plot(paths, prototypes, edges, deg, name_by_id) -> None:
    if not plotting.available():
        return
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception:
        return
    top = [pid for pid, _ in deg.most_common(8)]
    top_set = set(top)
    frags = set()
    B = nx.Graph()
    for e in edges:
        if e["prototype_id"] in top_set:
            B.add_node(e["prototype_id"], bip=0)
            B.add_node(e["fragment_id"], bip=1)
            B.add_edge(e["prototype_id"], e["fragment_id"])
            frags.add(e["fragment_id"])
    if B.number_of_nodes() == 0:
        return
    proto_nodes = [n for n in B.nodes if B.nodes[n]["bip"] == 0]
    frag_nodes = [n for n in B.nodes if B.nodes[n]["bip"] == 1]
    pos = {}
    for i, n in enumerate(proto_nodes):
        pos[n] = (0, -i * (len(frag_nodes) / max(1, len(proto_nodes))))
    for i, n in enumerate(frag_nodes):
        pos[n] = (1, -i)
    fig, ax = plt.subplots(figsize=(10, max(5, len(frag_nodes) * 0.35)))
    nx.draw_networkx_nodes(B, pos, nodelist=proto_nodes, node_color="#4C72B0",
                           node_size=800, ax=ax)
    nx.draw_networkx_nodes(B, pos, nodelist=frag_nodes, node_color="#DD8452",
                           node_size=180, ax=ax)
    nx.draw_networkx_edges(B, pos, alpha=0.3, ax=ax)
    labels = {n: name_by_id.get(n, n) for n in proto_nodes}
    nx.draw_networkx_labels(B, pos, labels, font_size=7, ax=ax)
    ax.set_title("Qspace prototype-fragment bipartite graph (top prototypes)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(paths.plots / "qspace_fragment_bipartite_graph.png", dpi=110)
    plt.close(fig)


def _intent_graph_plot(paths, prototypes, proximity_rows) -> None:
    if not plotting.available():
        return
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception:
        return
    G = nx.Graph()
    intents = Counter(p["intent"] for p in prototypes)
    for it, c in intents.items():
        G.add_node(it, size=c)
    name_intent = {p["prototype_id"]: p["intent"] for p in prototypes}
    for r in proximity_rows:
        ia = name_intent.get(r["prototype_a"])
        ib = name_intent.get(r["prototype_b"])
        if ia and ib and ia != ib and r["similarity"] >= 0.4:
            if G.has_edge(ia, ib):
                G[ia][ib]["weight"] += r["similarity"]
            else:
                G.add_edge(ia, ib, weight=r["similarity"])
    if G.number_of_nodes() == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(G, seed=7, k=0.8)
    sizes = [300 + intents[n] * 120 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#55A868", ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    ax.set_title("Qspace intent graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(paths.plots / "qspace_intent_graph.png", dpi=110)
    plt.close(fig)
