"""Components 10 & 11: evidence-role extraction and co-demand matrix."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from . import plotting
from .config import Paths
from .utils import (banner, jaccard, log, sub, truncate, write_csv, write_jsonl)


def extract_evidence(paths: Paths, affordances: List[Dict],
                     edges: List[Dict], classes: List[Dict] | None = None) -> List[Dict]:
    banner("EVIDENCE", "Evidence-role extraction")
    # map fragment -> prototypes it supports
    frag_protos: Dict[str, List[str]] = defaultdict(list)
    for e in edges:
        if e["support_type"] == "direct":
            frag_protos[e["fragment_id"]].append(e["prototype_id"])

    records: List[Dict] = []
    counter = [0]
    role_counter: Counter[str] = Counter()
    for rec in affordances:
        fid = rec["fragment_id"]
        did = rec["document_id"]
        text = rec["fragment_text"]
        seen: set = set()
        for d in rec["directly_answerable"]:
            for role, value in d.get("role_values", {}).items():
                if value is None:
                    continue
                dedup = (role, str(value))
                if dedup in seen:
                    continue
                seen.add(dedup)
                counter[0] += 1
                value_str = str(value)
                span = value_str if value_str in text else \
                    (d["supporting_span"])
                records.append({
                    "evidence_id": f"ev_{counter[0]:05d}",
                    "fragment_id": fid,
                    "document_id": did,
                    "role": role,
                    "value": truncate(value_str, 80),
                    "source_span": truncate(span, 120),
                    "confidence": round(d["confidence"] * 0.95, 3),
                    "linked_question_prototypes": sorted(set(frag_protos.get(fid, []))),
                })
                role_counter[role] += 1

    write_jsonl(paths.out / "evidence_roles.jsonl", records)
    write_csv(paths.out / "evidence_roles.csv", records, columns=[
        "evidence_id", "fragment_id", "document_id", "role", "value",
        "source_span", "confidence", "linked_question_prototypes"])

    log("EVIDENCE", f"Evidence units extracted: {len(records)}")
    log("EVIDENCE", "Evidence role frequency:")
    for role, c in role_counter.most_common(12):
        sub(f"{role}: {c}")
    # example
    example_frag = None
    for rec in affordances:
        vals = [r for r in records if r["fragment_id"] == rec["fragment_id"]]
        if len(vals) >= 2:
            example_frag = (rec["fragment_id"], vals[:4])
            break
    if example_frag:
        log("EVIDENCE", f"{example_frag[0]}")
        for r in example_frag[1]:
            sub(f"{r['role']} = {r['value']}")

    plotting.bar(paths.plots / "evidence_role_frequency.png",
                 [r for r, _ in role_counter.most_common()],
                 [c for _, c in role_counter.most_common()],
                 "Evidence role frequency", ylabel="occurrences",
                 horizontal=True)
    _evidence_by_class_plot(paths, records, classes or [])
    log("EVIDENCE", f"Saved: {paths.out / 'evidence_roles.jsonl'}")
    return records


def _evidence_by_class_plot(paths: Paths, records: List[Dict],
                            classes: List[Dict]) -> None:
    if not classes or not records:
        return
    frag_class: Dict[str, str] = {}
    class_names: List[str] = []
    for c in classes:
        class_names.append(c["name"])
        for f in c.get("member_fragments", []):
            frag_class[f] = c["name"]
    roles = sorted({r["role"] for r in records})
    if not roles:
        return
    cidx = {n: i for i, n in enumerate(class_names)}
    ridx = {r: i for i, r in enumerate(roles)}
    mat = [[0.0] * len(roles) for _ in class_names]
    for r in records:
        cn = frag_class.get(r["fragment_id"])
        if cn in cidx:
            mat[cidx[cn]][ridx[r["role"]]] += 1
    plotting.heatmap(paths.plots / "evidence_roles_by_class.png",
                     mat, class_names, roles,
                     "Evidence roles by class (counts)", cmap="YlOrBr",
                     annotate=len(class_names) * len(roles) <= 200)


def compute_codemand(paths: Paths, prototypes: List[Dict],
                     density_by_proto: Dict[str, float]) -> List[Dict]:
    """S(e_a, e_b) = sum_q p(q) * I(e_a in q) * I(e_b in q)."""
    banner("CODEMAND", "Evidence co-demand matrix")
    roles: List[str] = []
    for p in prototypes:
        for r in p["required_evidence_roles"]:
            if r not in roles:
                roles.append(r)
    idx = {r: i for i, r in enumerate(roles)}
    n = len(roles)
    mat = [[0.0] * n for _ in range(n)]

    total_p = sum(density_by_proto.get(p["prototype_id"], 0.0)
                  for p in prototypes) or 1.0
    for p in prototypes:
        pq = density_by_proto.get(p["prototype_id"], 0.0) / total_p
        rr = p["required_evidence_roles"]
        for a in rr:
            for b in rr:
                mat[idx[a]][idx[b]] += pq

    rows: List[Dict] = []
    pairs: List[Tuple[str, str, float]] = []
    for i in range(n):
        for j in range(n):
            rows.append({"role_a": roles[i], "role_b": roles[j],
                         "co_demand": round(mat[i][j], 4)})
            if i < j and mat[i][j] > 0:
                pairs.append((roles[i], roles[j], mat[i][j]))
    write_csv(paths.out / "evidence_codemand.csv", rows,
              columns=["role_a", "role_b", "co_demand"])

    pairs.sort(key=lambda t: -t[2])
    log("CODEMAND", "Strong evidence co-demand pairs:")
    for a, b, v in pairs[:8]:
        sub(f"{a} <-> {b}: {v:.3f}")

    if n:
        maxv = max((mat[i][j] for i in range(n) for j in range(n)), default=1.0) or 1.0
        norm = [[mat[i][j] / maxv for j in range(n)] for i in range(n)]
        plotting.heatmap(paths.plots / "evidence_codemand_heatmap.png",
                         norm, roles, roles,
                         "Evidence co-demand (normalized)", cmap="magma",
                         annotate=n <= 16)
    log("CODEMAND", f"Saved: {paths.out / 'evidence_codemand.csv'}")
    return rows
