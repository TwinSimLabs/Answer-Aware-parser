"""Component 13: retrieval objective planner (query-time simulation)."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .config import NUM_RETRIEVAL_SIMULATIONS, Paths
from .utils import banner, log, sub, truncate, write_csv, write_jsonl


def _traverse(tree: Dict, target_roles: List[str]) -> Tuple[List[str], int, int]:
    """Return (ordered traversal path, nodes_visited, context_tokens)."""
    node_map = {n["node_id"]: n for n in tree["nodes"]}
    children = {n["node_id"]: n["children"] for n in tree["nodes"]}
    parent = {}
    for n in tree["nodes"]:
        for c in n["children"]:
            parent[c] = n["node_id"]

    role_leaf = {n["role"]: n["node_id"] for n in tree["nodes"]
                 if n.get("role")}
    visited: List[str] = []
    seen = set()
    tokens = 0

    def path_to(nid):
        p = [nid]
        cur = parent.get(nid)
        while cur is not None:
            p.append(cur)
            cur = parent.get(cur)
        return list(reversed(p))

    for role in target_roles:
        leaf = role_leaf.get(role)
        if leaf is None:
            continue
        for nd in path_to(leaf):
            if nd not in seen:
                seen.add(nd)
                visited.append(nd)
                tokens += 20  # approx per-node context cost
    if not visited and "root" in node_map:
        visited = ["root"]
        tokens = 20
    return visited, len(visited), tokens


def simulate_queries(paths: Paths, prototypes: List[Dict], edges: List[Dict],
                     trees: List[Dict], evidence: List[Dict],
                     proto_to_class: Dict[str, str],
                     affordances: List[Dict]) -> List[Dict]:
    banner("PLANNER", "Retrieval objective planner (query simulation)")

    tree_by_frag = {t["fragment_id"]: t for t in trees}
    proto_by_id = {p["prototype_id"]: p for p in prototypes}
    frag_evs: Dict[str, List[Dict]] = defaultdict(list)
    for ev in evidence:
        frag_evs[ev["fragment_id"]].append(ev)

    # build query bank from direct affordances that map to a compiled tree
    edge_lookup: Dict[Tuple[str, str], Dict] = {}
    for e in edges:
        edge_lookup[(e["fragment_id"], e["prototype_id"])] = e

    # map (fragment,intent/slots) -> prototype via edges
    frag_proto_map: Dict[str, List[str]] = defaultdict(list)
    for e in edges:
        if e["support_type"] == "direct":
            frag_proto_map[e["fragment_id"]].append(e["prototype_id"])

    queries: List[Dict] = []
    for rec in affordances:
        fid = rec["fragment_id"]
        if fid not in tree_by_frag:
            continue
        for d in rec["directly_answerable"]:
            # find prototype id for this affordance
            pids = frag_proto_map.get(fid, [])
            match = None
            for pid in pids:
                if proto_by_id.get(pid, {}).get("intent") == d["intent"]:
                    match = pid
                    break
            if not match:
                continue
            queries.append({"query": d["raw_question"], "fragment_id": fid,
                            "prototype_id": match, "slots": d["slots"],
                            "target_evidence_roles": d["evidence_roles"],
                            "gold_span": d["supporting_span"]})

    sims: List[Dict] = []
    for q in queries[:NUM_RETRIEVAL_SIMULATIONS]:
        tree = tree_by_frag[q["fragment_id"]]
        path, n_visited, tokens = _traverse(tree, q["target_evidence_roles"])
        node_map = {n["node_id"]: n for n in tree["nodes"]}
        answer_evidence = []
        for nid in path:
            n = node_map.get(nid, {})
            if n.get("role") in q["target_evidence_roles"]:
                answer_evidence.append({
                    "role": n["role"],
                    "value": n["label"].split(": ", 1)[-1],
                    "source_span": truncate(str(n.get("span") or ""), 120)})
        sims.append({
            "query": q["query"],
            "matched_prototype_id": q["prototype_id"],
            "class_id": proto_to_class.get(q["prototype_id"], ""),
            "slots": q["slots"],
            "target_evidence_roles": q["target_evidence_roles"],
            "selected_tree_id": tree["tree_id"],
            "traversal_path": path,
            "nodes_visited": n_visited,
            "context_tokens_loaded": tokens,
            "answer_evidence": answer_evidence,
        })

    write_jsonl(paths.out / "retrieval_simulations.jsonl", sims)
    write_csv(paths.out / "retrieval_simulations.csv", [{
        "query": s["query"], "matched_prototype_id": s["matched_prototype_id"],
        "class_id": s["class_id"], "selected_tree_id": s["selected_tree_id"],
        "nodes_visited": s["nodes_visited"],
        "context_tokens_loaded": s["context_tokens_loaded"],
        "traversal_path": s["traversal_path"]} for s in sims])

    log("PLANNER", f"Simulated {len(sims)} query traversals.")
    for s in sims[:3]:
        sub(f"Q: {s['query']}")
        sub(f"   tree={s['selected_tree_id']} nodes_visited={s['nodes_visited']} "
            f"tokens={s['context_tokens_loaded']}", 4)
        if s["answer_evidence"]:
            ev = s["answer_evidence"][0]
            sub(f"   answer: {ev['role']} = {ev['value']}", 4)

    _write_trace_report(paths, sims)
    log("PLANNER", f"Saved: {paths.out / 'retrieval_simulations.jsonl'}")
    return sims


def _write_trace_report(paths: Paths, sims: List[Dict]) -> None:
    lines = ["# Sample Retrieval Traces", "",
             f"Showing {min(len(sims), 20)} of {len(sims)} simulated "
             "query traversals over compiled Qspace trees.", ""]
    for i, s in enumerate(sims[:20], start=1):
        lines += [f"## {i}. {s['query']}", "",
                  f"- matched prototype: `{s['matched_prototype_id']}`",
                  f"- class: `{s['class_id']}`",
                  f"- selected tree: `{s['selected_tree_id']}`",
                  f"- target evidence roles: {s['target_evidence_roles']}",
                  f"- traversal path: {' -> '.join(s['traversal_path'])}",
                  f"- nodes visited: {s['nodes_visited']} | context tokens: "
                  f"{s['context_tokens_loaded']}", "", "Answer evidence:", ""]
        for ev in s["answer_evidence"]:
            lines.append(f"  - **{ev['role']}** = {ev['value']}  "
                         f"(span: \"{ev['source_span']}\")")
        if not s["answer_evidence"]:
            lines.append("  - (no direct evidence leaf reached)")
        lines.append("")
    (paths.reports / "sample_retrieval_traces.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
