"""Component 12: retrieval tree compiler.

For each fragment we generate five candidate trees, score each with an explicit
cost model, and select the tree minimizing:

    J(T) = sum_q p(q|f) * Cost(q, T)
           + lambda * tree_complexity(T)
           - mu     * trace_clarity(T)
           - nu     * citation_locality(T)

The LLM (if present) could *propose* trees, but selection is always driven by
these transparent metrics.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from . import plotting
from .config import CostWeights, MAX_TREE_VISUALS, Paths
from .utils import (banner, estimate_tokens, log, sub, truncate, write_csv,
                    write_jsonl)

WEIGHTS = CostWeights()

# ---------------------------------------------------------------------------
# Internal tree representation helpers.
# ---------------------------------------------------------------------------
class _Tree:
    def __init__(self, candidate_type: str):
        self.candidate_type = candidate_type
        self.nodes: Dict[str, Dict] = {}
        self.role_to_node: Dict[str, str] = {}
        self.add("root", "Root", None, tokens=4)

    def add(self, node_id: str, label: str, parent: Optional[str],
            tokens: int = 6, role: Optional[str] = None,
            evidence_id: Optional[str] = None, span: Optional[str] = None) -> str:
        self.nodes[node_id] = {
            "node_id": node_id, "label": label, "parent": parent,
            "children": [], "tokens": tokens, "role": role,
            "evidence_id": evidence_id, "span": span}
        if parent is not None:
            self.nodes[parent]["children"].append(node_id)
        if role is not None:
            self.role_to_node[role] = node_id
        return node_id

    def depth(self, node_id: str) -> int:
        d = 0
        cur = self.nodes[node_id]["parent"]
        while cur is not None:
            d += 1
            cur = self.nodes[cur]["parent"]
        return d

    def path(self, node_id: str) -> List[str]:
        p = [node_id]
        cur = self.nodes[node_id]["parent"]
        while cur is not None:
            p.append(cur)
            cur = self.nodes[cur]["parent"]
        return list(reversed(p))

    def max_depth(self) -> int:
        return max((self.depth(n) for n in self.nodes), default=0)

    def leaves(self) -> List[str]:
        return [n for n, d in self.nodes.items() if not d["children"]]

    def top_branch(self, node_id: str) -> str:
        p = self.path(node_id)
        return p[1] if len(p) > 1 else node_id


# ---------------------------------------------------------------------------
# Candidate tree builders.
# ---------------------------------------------------------------------------
def _evidence_label(ev: Dict) -> str:
    return f"{ev['role']}: {truncate(str(ev['value']), 40)}"


def _build_source_order(evs: List[Dict]) -> _Tree:
    """Linear chain mirroring document order (deep, one branch)."""
    t = _Tree("source_order_tree")
    prev = "root"
    for i, ev in enumerate(evs):
        nid = f"n{i}"
        t.add(nid, _evidence_label(ev), prev,
              tokens=estimate_tokens(str(ev["value"])) + 6,
              role=ev["role"], evidence_id=ev["evidence_id"],
              span=ev["source_span"])
        prev = nid
    return t


def _build_evidence_role(evs: List[Dict]) -> _Tree:
    """root -> role-group -> evidence leaves (depth 2)."""
    t = _Tree("evidence_role_tree")
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for ev in evs:
        groups[ev["role"]].append(ev)
    for gi, (role, items) in enumerate(groups.items()):
        gid = f"g{gi}"
        t.add(gid, f"Role: {role}", "root", tokens=5)
        for k, ev in enumerate(items):
            t.add(f"{gid}_{k}", _evidence_label(ev), gid,
                  tokens=estimate_tokens(str(ev["value"])) + 4,
                  role=ev["role"], evidence_id=ev["evidence_id"],
                  span=ev["source_span"])
    return t


def _build_question_intent(evs: List[Dict], proto_roles: Dict[str, List[str]],
                           footprint: List[Tuple[str, float, str]]) -> _Tree:
    """root -> intent -> evidence leaves needed by that intent."""
    t = _Tree("question_intent_tree")
    role_evs: Dict[str, List[Dict]] = defaultdict(list)
    for ev in evs:
        role_evs[ev["role"]].append(ev)
    placed = set()
    for ii, (pid, w, intent) in enumerate(footprint):
        iid = f"i{ii}"
        t.add(iid, f"Intent: {intent}", "root", tokens=5)
        for role in proto_roles.get(pid, []):
            for k, ev in enumerate(role_evs.get(role, [])):
                key = ev["evidence_id"]
                nid = f"{iid}_{role}_{k}"
                t.add(nid, _evidence_label(ev), iid,
                      tokens=estimate_tokens(str(ev["value"])) + 4,
                      role=ev["role"], evidence_id=ev["evidence_id"],
                      span=ev["source_span"])
                placed.add(key)
    # attach any orphan evidence under root
    for k, ev in enumerate(evs):
        if ev["evidence_id"] not in placed and ev["role"] not in t.role_to_node:
            t.add(f"orphan{k}", _evidence_label(ev), "root",
                  tokens=estimate_tokens(str(ev["value"])) + 4,
                  role=ev["role"], evidence_id=ev["evidence_id"],
                  span=ev["source_span"])
    return t


def _build_class_objective(evs: List[Dict], class_objectives: List[str],
                           role_to_intent: Dict[str, str]) -> _Tree:
    t = _Tree("class_objective_tree")
    obj_node: Dict[str, str] = {}
    for oi, obj in enumerate(class_objectives):
        obj_node[obj] = t.add(f"o{oi}", f"Objective: {obj}", "root", tokens=5)
    misc = t.add("omisc", "Objective: other", "root", tokens=5)
    for k, ev in enumerate(evs):
        intent = role_to_intent.get(ev["role"], "")
        parent = obj_node.get(intent, misc)
        t.add(f"e{k}", _evidence_label(ev), parent,
              tokens=estimate_tokens(str(ev["value"])) + 4,
              role=ev["role"], evidence_id=ev["evidence_id"],
              span=ev["source_span"])
    # prune empty objective nodes
    for nid in list(t.nodes):
        nd = t.nodes[nid]
        if nid not in ("root",) and nd["label"].startswith("Objective") \
                and not nd["children"]:
            t.nodes["root"]["children"].remove(nid)
            del t.nodes[nid]
    return t


def _build_hybrid(evs: List[Dict], role_density: Dict[str, float],
                  codemand: Dict[Tuple[str, str], float]) -> _Tree:
    """High-density evidence near root; co-demanded roles grouped together."""
    t = _Tree("hybrid_optimized_tree")
    # order roles by density desc
    roles = list({ev["role"] for ev in evs})
    roles.sort(key=lambda r: -role_density.get(r, 0.0))
    role_evs: Dict[str, List[Dict]] = defaultdict(list)
    for ev in evs:
        role_evs[ev["role"]].append(ev)

    grouped: List[List[str]] = []
    used = set()
    for r in roles:
        if r in used:
            continue
        cluster = [r]
        used.add(r)
        for r2 in roles:
            if r2 in used:
                continue
            if codemand.get((r, r2), 0.0) >= 0.05:
                cluster.append(r2)
                used.add(r2)
        grouped.append(cluster)

    # place highest-density cluster shallow; each cluster is a compact branch
    for gi, cluster in enumerate(grouped):
        if len(cluster) == 1:
            # attach evidence directly to root for max shallowness
            for k, ev in enumerate(role_evs[cluster[0]]):
                t.add(f"h{gi}_{k}", _evidence_label(ev), "root",
                      tokens=estimate_tokens(str(ev["value"])) + 4,
                      role=ev["role"], evidence_id=ev["evidence_id"],
                      span=ev["source_span"])
        else:
            gid = t.add(f"hg{gi}", "Co-demanded: " + ", ".join(cluster),
                        "root", tokens=5)
            for r in cluster:
                for k, ev in enumerate(role_evs[r]):
                    t.add(f"{gid}_{r}_{k}", _evidence_label(ev), gid,
                          tokens=estimate_tokens(str(ev["value"])) + 4,
                          role=ev["role"], evidence_id=ev["evidence_id"],
                          span=ev["source_span"])
    return t


# ---------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------
def _score_tree(t: _Tree, footprint: List[Tuple[str, float, str]],
                proto_roles: Dict[str, List[str]]) -> Dict:
    total_w = sum(w for _, w, _ in footprint) or 1.0
    exp_path = exp_nodes = exp_tokens = exp_reason = 0.0
    exp_cross = exp_missing = exp_ambig = 0.0
    citation_local = 0.0

    for pid, w, intent in footprint:
        wn = w / total_w
        req_roles = proto_roles.get(pid, [])
        leaf_nodes = [t.role_to_node[r] for r in req_roles if r in t.role_to_node]
        missing = sum(1 for r in req_roles if r not in t.role_to_node)
        req = max(1, len(req_roles))
        if leaf_nodes:
            visited = set()
            depths = []
            branches = set()
            tok = 0
            for ln in leaf_nodes:
                pth = t.path(ln)
                for nd in pth:
                    if nd not in visited:
                        visited.add(nd)
                        tok += t.nodes[nd]["tokens"]
                depths.append(t.depth(ln))
                branches.add(t.top_branch(ln))
            exp_path += wn * (sum(depths) / len(depths))
            exp_nodes += wn * len(visited)
            exp_tokens += wn * tok
            exp_reason += wn * max(0, len(visited) - len(leaf_nodes))
            exp_cross += wn * max(0, len(branches) - 1)
            # citation locality: leaves for same q close together
            if len(leaf_nodes) > 1:
                avg_depth = sum(depths) / len(depths)
                spread = (max(depths) - min(depths)) + (len(branches) - 1)
                citation_local += wn * (1.0 / (1.0 + spread))
            else:
                citation_local += wn * 1.0
        else:
            exp_path += wn * (t.max_depth() + 1)
            exp_tokens += wn * sum(n["tokens"] for n in t.nodes.values())
            exp_nodes += wn * len(t.nodes)
        exp_missing += wn * (missing / req)
        # ambiguity: multiple evidence leaves share a role
        role_counts = Counter(n["role"] for n in t.nodes.values() if n["role"])
        amb = sum(1 for r in req_roles if role_counts.get(r, 0) > 1)
        exp_ambig += wn * (amb / req)

    n_nodes = len(t.nodes)
    depth = t.max_depth()
    tree_complexity = (n_nodes + depth) / 20.0
    trace_complexity = depth / 6.0
    trace_clarity = 1.0 / (1.0 + trace_complexity)
    citation_distance = 1.0 - citation_local

    # normalized components
    norm_tokens = exp_tokens / 200.0
    total_cost = (
        WEIGHTS.path_length * (exp_path / 5.0)
        + WEIGHTS.nodes_visited * (exp_nodes / 8.0)
        + WEIGHTS.context_tokens * norm_tokens
        + WEIGHTS.reasoning_steps * (exp_reason / 5.0)
        + WEIGHTS.ambiguity * exp_ambig
        + WEIGHTS.missing_evidence * exp_missing
        + WEIGHTS.cross_branch * (exp_cross / 3.0)
        + WEIGHTS.trace_complexity * trace_complexity
        + WEIGHTS.citation_distance * citation_distance
        + WEIGHTS.tree_complexity * tree_complexity)

    j = (total_cost
         + WEIGHTS.lambda_complexity * tree_complexity
         - WEIGHTS.mu_trace_clarity * trace_clarity
         - WEIGHTS.nu_citation_locality * citation_local)

    return {
        "expected_path_length": round(exp_path, 3),
        "expected_nodes_visited": round(exp_nodes, 3),
        "expected_context_tokens": round(exp_tokens, 1),
        "expected_reasoning_steps": round(exp_reason, 3),
        "ambiguity_penalty": round(exp_ambig, 3),
        "cross_branch_jump_penalty": round(exp_cross, 3),
        "missing_evidence_penalty": round(exp_missing, 3),
        "trace_complexity": round(trace_complexity, 3),
        "citation_locality": round(citation_local, 3),
        "citation_distance": round(citation_distance, 3),
        "trace_clarity": round(trace_clarity, 3),
        "tree_complexity": round(tree_complexity, 3),
        "total_cost": round(total_cost, 4),
        "objective_J": round(j, 4),
        "num_nodes": n_nodes,
        "max_depth": depth,
    }


def _export_nodes(t: _Tree) -> List[Dict]:
    out = []
    for nid, nd in t.nodes.items():
        out.append({
            "node_id": nid, "label": nd["label"],
            "children": nd["children"],
            "evidence_id": nd["evidence_id"], "role": nd["role"],
            "span": nd["span"]})
    return out


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def compile_trees(paths: Paths, fragments: List[Dict], edges: List[Dict],
                  evidence: List[Dict], prototypes: List[Dict],
                  density: Dict[str, float], classes: List[Dict],
                  proto_to_class: Dict[str, str],
                  codemand_rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    banner("TREE", "Retrieval tree compiler")

    proto_by_id = {p["prototype_id"]: p for p in prototypes}
    proto_roles = {p["prototype_id"]: p["required_evidence_roles"]
                   for p in prototypes}
    role_to_intent: Dict[str, str] = {}
    for p in prototypes:
        for r in p["required_evidence_roles"]:
            role_to_intent.setdefault(r, p["intent"])
    class_by_id = {c["class_id"]: c for c in classes}

    # role density = mean density of prototypes requiring that role
    role_density: Dict[str, List[float]] = defaultdict(list)
    for p in prototypes:
        for r in p["required_evidence_roles"]:
            role_density[r].append(density.get(p["prototype_id"], 0.0))
    role_density_mean = {r: (sum(v) / len(v) if v else 0.0)
                         for r, v in role_density.items()}

    codemand = {(r["role_a"], r["role_b"]): r["co_demand"] for r in codemand_rows}

    frag_evs: Dict[str, List[Dict]] = defaultdict(list)
    for ev in evidence:
        frag_evs[ev["fragment_id"]].append(ev)
    frag_protos: Dict[str, List[str]] = defaultdict(list)
    for e in edges:
        if e["support_type"] == "direct":
            frag_protos[e["fragment_id"]].append(e["prototype_id"])
    frag_by_id = {f["fragment_id"]: f for f in fragments}

    all_trees: List[Dict] = []
    candidate_rows: List[Dict] = []
    selected_rows: List[Dict] = []
    selected_type_counter: Counter[str] = Counter()
    cost_by_type: Dict[str, List[float]] = defaultdict(list)
    path_by_type: Dict[str, List[float]] = defaultdict(list)
    tokens_by_type: Dict[str, List[float]] = defaultdict(list)

    target_fragments = [fid for fid in frag_evs
                        if frag_evs[fid] and frag_protos.get(fid)]

    for fid in target_fragments:
        evs = frag_evs[fid]
        protos = list(dict.fromkeys(frag_protos[fid]))
        # footprint with p(q|f)
        weights = [density.get(p, 0.01) for p in protos]
        tot = sum(weights) or 1.0
        footprint = [(p, w / tot, proto_by_id[p]["intent"])
                     for p, w in zip(protos, weights)]
        cid = proto_to_class.get(protos[0], "")
        objectives = class_by_id.get(cid, {}).get("class_objectives", [])

        candidates = [
            _build_source_order(evs),
            _build_evidence_role(evs),
            _build_question_intent(evs, proto_roles, footprint),
            _build_class_objective(evs, objectives or [role_to_intent.get(e["role"], "other") for e in evs], role_to_intent),
            _build_hybrid(evs, role_density_mean, codemand),
        ]
        scored: List[Tuple[_Tree, Dict]] = []
        for t in candidates:
            sc = _score_tree(t, footprint, proto_roles)
            scored.append((t, sc))
            cost_by_type[t.candidate_type].append(sc["total_cost"])
            path_by_type[t.candidate_type].append(sc["expected_path_length"])
            tokens_by_type[t.candidate_type].append(sc["expected_context_tokens"])
            candidate_rows.append({
                "fragment_id": fid, "candidate_type": t.candidate_type,
                **{k: sc[k] for k in (
                    "expected_path_length", "expected_nodes_visited",
                    "expected_context_tokens", "expected_reasoning_steps",
                    "ambiguity_penalty", "cross_branch_jump_penalty",
                    "missing_evidence_penalty", "trace_complexity",
                    "citation_locality", "trace_clarity", "tree_complexity",
                    "total_cost", "objective_J", "num_nodes", "max_depth")}})

        best_tree, best_score = min(scored, key=lambda ts: ts[1]["objective_J"])
        selected_type_counter[best_tree.candidate_type] += 1
        frag_text = frag_by_id.get(fid, {}).get("text", "")
        tree_record = {
            "tree_id": f"tree_{fid}_v1",
            "fragment_id": fid,
            "class_id": cid,
            "optimized_for": sorted({intent for _, _, intent in footprint}),
            "selected_candidate_type": best_tree.candidate_type,
            "nodes": _export_nodes(best_tree),
            "score": best_score,
            "candidate_scores": {t.candidate_type: sc["objective_J"]
                                 for t, sc in scored},
            "provenance": {"source_fragment_id": fid,
                           "source_document_id": frag_by_id.get(fid, {}).get("document_id", ""),
                           "fragment_text": truncate(frag_text, 200)},
        }
        all_trees.append(tree_record)
        selected_rows.append({
            "tree_id": tree_record["tree_id"], "fragment_id": fid,
            "class_id": cid,
            "selected_candidate_type": best_tree.candidate_type,
            "optimized_for": tree_record["optimized_for"],
            **{k: best_score[k] for k in (
                "expected_path_length", "expected_context_tokens",
                "trace_clarity", "citation_locality", "total_cost",
                "objective_J")}})

    write_jsonl(paths.out / "retrieval_trees.jsonl", all_trees)
    write_csv(paths.out / "candidate_tree_scores.csv", candidate_rows)
    write_csv(paths.out / "selected_tree_scores.csv", selected_rows)

    log("TREE", f"Fragments compiled into trees: {len(all_trees)}")
    log("TREE", "Selected tree type distribution:")
    for ttype, c in selected_type_counter.most_common():
        sub(f"{ttype}: {c}")
    log("TREE", "Average total_cost by candidate type:")
    for ttype in ("source_order_tree", "evidence_role_tree",
                  "question_intent_tree", "class_objective_tree",
                  "hybrid_optimized_tree"):
        vals = cost_by_type.get(ttype, [])
        if vals:
            sub(f"{ttype}: {sum(vals) / len(vals):.3f}")

    # example scorecard
    if all_trees:
        ex = all_trees[0]
        log("TREE", f"Fragment {ex['fragment_id']} candidate scorecard (objective J):")
        for ttype, j in ex["candidate_scores"].items():
            sub(f"{ttype}: J={j}")
        sub(f"SELECTED: {ex['selected_candidate_type']}")

    _tree_plots(paths, cost_by_type, path_by_type, tokens_by_type,
                selected_type_counter)
    _render_tree_visuals(paths, all_trees, density, proto_by_id, frag_protos)
    log("TREE", f"Saved: {paths.out / 'retrieval_trees.jsonl'}")
    return all_trees, selected_rows


def _tree_plots(paths, cost_by_type, path_by_type, tokens_by_type,
                selected_counter) -> None:
    types = ["source_order_tree", "evidence_role_tree", "question_intent_tree",
             "class_objective_tree", "hybrid_optimized_tree"]

    def mean(d, k):
        return sum(d.get(k, [0])) / max(1, len(d.get(k, [1])))

    present = [t for t in types if cost_by_type.get(t)]
    plotting.bar(paths.plots / "candidate_tree_cost_comparison.png",
                 present, [mean(cost_by_type, t) for t in present],
                 "Mean total_cost by candidate tree type", ylabel="total_cost")
    plotting.bar(paths.plots / "expected_path_length_by_tree_type.png",
                 present, [mean(path_by_type, t) for t in present],
                 "Mean expected path length by tree type",
                 ylabel="path length", color="#DD8452")
    plotting.bar(paths.plots / "context_tokens_by_tree_type.png",
                 present, [mean(tokens_by_type, t) for t in present],
                 "Mean expected context tokens by tree type",
                 ylabel="tokens", color="#55A868")
    plotting.bar(paths.plots / "selected_tree_type_distribution.png",
                 [t for t, _ in selected_counter.most_common()],
                 [c for _, c in selected_counter.most_common()],
                 "Selected tree type distribution", ylabel="fragments",
                 color="#8172B3")


def _dot(tree: Dict) -> str:
    lines = ["digraph T {", '  rankdir=TB;',
             '  node [shape=box, style=rounded, fontsize=10];']
    label_by = {n["node_id"]: n["label"] for n in tree["nodes"]}
    for n in tree["nodes"]:
        safe = n["label"].replace('"', "'")
        shape = "ellipse" if n["evidence_id"] else "box"
        lines.append(f'  "{n["node_id"]}" [label="{safe}", shape={shape}];')
    for n in tree["nodes"]:
        for ch in n["children"]:
            lines.append(f'  "{n["node_id"]}" -> "{ch}";')
    lines.append("}")
    return "\n".join(lines)


def _render_tree_visuals(paths, all_trees, density, proto_by_id,
                         frag_protos) -> None:
    # pick representative trees: highest total footprint density
    def frag_density(t):
        return sum(density.get(p, 0.0) for p in frag_protos.get(t["fragment_id"], []))
    reps = sorted(all_trees, key=frag_density, reverse=True)[:MAX_TREE_VISUALS]
    for tr in reps:
        fid = tr["fragment_id"]
        dot = _dot(tr)
        (paths.trees / f"tree_{fid}.dot").write_text(dot, encoding="utf-8")
        _render_png(paths, tr, fid)
        _render_html(paths, tr, fid)
    log("TREE", f"Rendered {len(reps)} representative tree visuals in "
                f"{paths.trees}")


def _render_png(paths, tree, fid) -> None:
    if not plotting.available():
        return
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception:
        return
    G = nx.DiGraph()
    labels = {}
    is_ev = {}
    for n in tree["nodes"]:
        G.add_node(n["node_id"])
        labels[n["node_id"]] = n["label"]
        is_ev[n["node_id"]] = bool(n["evidence_id"])
    for n in tree["nodes"]:
        for ch in n["children"]:
            G.add_edge(n["node_id"], ch)
    pos = _hierarchy_pos(G, "root")
    fig, ax = plt.subplots(figsize=(max(8, len(G) * 0.7), 6))
    node_colors = ["#DD8452" if is_ev[n] else "#4C72B0" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1600,
                           ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True)
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    ax.set_title(f"Retrieval tree {tree['tree_id']} "
                 f"({tree['selected_candidate_type']})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(paths.trees / f"tree_{fid}.png", dpi=110)
    plt.close(fig)


def _hierarchy_pos(G, root, width=1.0, vert_gap=0.25, xcenter=0.5,
                   pos=None, depth=0):
    if pos is None:
        pos = {}
    pos[root] = (xcenter, -depth * vert_gap)
    children = list(G.successors(root)) if hasattr(G, "successors") else []
    if children:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for ch in children:
            nextx += dx
            _hierarchy_pos(G, ch, dx, vert_gap, nextx, pos, depth + 1)
    return pos


def _render_html(paths, tree, fid) -> None:
    def render_node(nid, node_map):
        n = node_map[nid]
        tag = "evidence" if n["evidence_id"] else "branch"
        html = f'<li class="{tag}"><span>{n["label"]}</span>'
        if n["span"]:
            html += f'<div class="span">"{truncate(str(n["span"]), 100)}"</div>'
        if n["children"]:
            html += "<ul>" + "".join(
                render_node(c, node_map) for c in n["children"]) + "</ul>"
        return html + "</li>"

    node_map = {n["node_id"]: n for n in tree["nodes"]}
    body = render_node("root", node_map) if "root" in node_map else ""
    sc = tree["score"]
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>{tree['tree_id']}</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222}}
ul{{list-style:none}} li{{margin:4px 0;padding-left:8px;border-left:2px solid #ccc}}
.branch>span{{background:#4C72B0;color:#fff;padding:2px 8px;border-radius:6px}}
.evidence>span{{background:#DD8452;color:#fff;padding:2px 8px;border-radius:6px}}
.span{{color:#666;font-size:12px;margin:2px 0 2px 6px}}
.card{{background:#f5f5f5;padding:10px;border-radius:8px;margin-bottom:14px}}
</style></head><body>
<h2>{tree['tree_id']} <small>({tree['selected_candidate_type']})</small></h2>
<div class='card'>optimized_for: {', '.join(tree['optimized_for'])}<br>
expected_path_length={sc['expected_path_length']} |
expected_context_tokens={sc['expected_context_tokens']} |
trace_clarity={sc['trace_clarity']} | total_cost={sc['total_cost']} |
J={sc['objective_J']}</div>
<ul>{body}</ul></body></html>"""
    (paths.trees / f"tree_{fid}.html").write_text(html, encoding="utf-8")
