"""Component 5: question prototype normalization.

Raw questions are grouped deterministically into canonical prototypes by
(intent, slot-signature, required-evidence-role-signature). The grouping logic
is fully inspectable; an LLM would only be used to prettify names.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from . import plotting
from .config import Paths
from .utils import banner, log, sub, write_csv, write_json


def _slot_signature(slots: Dict) -> Tuple[str, ...]:
    return tuple(sorted(k for k in slots.keys()))


def _canonical_name(intent: str, slot_sig: Tuple[str, ...]) -> str:
    if slot_sig:
        return f"{intent}({', '.join(slot_sig)})"
    return f"{intent}()"


_CANONICAL_QUESTION = {
    "approval_lookup": "Who approves a transaction under the given conditions?",
    "threshold_lookup": "What threshold applies under the given conditions?",
    "owner_lookup": "Who owns this object?",
    "effective_date_lookup": "When does this become effective?",
    "applicability_check": "Does this apply to the given audience?",
    "exception_lookup": "What is the exception to this rule?",
    "rule_lookup": "What rule applies to the given subject?",
    "metric_lookup": "What is the value of this metric for the period?",
    "variance_explanation": "What explains the variance for this metric?",
    "sla_lookup": "What is the SLA for this step?",
    "next_step_lookup": "What is the next step and who owns it?",
    "escalation_lookup": "Who handles escalation when the SLA is missed?",
}


def normalize_prototypes(paths: Paths, affordances: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    banner("QPROTO", "Question prototype normalization")

    groups: Dict[Tuple, Dict] = {}
    edges: List[Dict] = []
    raw_count = 0

    for rec in affordances:
        for d in rec["directly_answerable"]:
            raw_count += 1
            intent = d["intent"]
            slot_sig = _slot_signature(d["slots"])
            role_sig = tuple(sorted(d["evidence_roles"]))
            key = (intent, slot_sig, role_sig)
            g = groups.setdefault(key, {
                "prototype_id": None,
                "intent": intent,
                "slot_sig": slot_sig,
                "role_sig": role_sig,
                "canonical_name": _canonical_name(intent, slot_sig),
                "canonical_question": _CANONICAL_QUESTION.get(
                    intent, f"What does this fragment answer about {intent}?"),
                "answer_shape": d["answer_shape"],
                "example_raw_questions": [],
                "support_fragment_ids": set(),
                "support_document_ids": set(),
            })
            if len(g["example_raw_questions"]) < 8 and \
                    d["raw_question"] not in g["example_raw_questions"]:
                g["example_raw_questions"].append(d["raw_question"])
            g["support_fragment_ids"].add(rec["fragment_id"])
            g["support_document_ids"].add(rec["document_id"])

    # assign ids ordered by support size
    ordered = sorted(groups.values(),
                     key=lambda g: -len(g["support_fragment_ids"]))
    prototypes: List[Dict] = []
    key_to_id: Dict[Tuple, str] = {}
    for i, g in enumerate(ordered, start=1):
        pid = f"qproto_{i:03d}_{g['intent']}"
        g["prototype_id"] = pid
        key = (g["intent"], g["slot_sig"], g["role_sig"])
        key_to_id[key] = pid
        prototypes.append({
            "prototype_id": pid,
            "canonical_name": g["canonical_name"],
            "canonical_question": g["canonical_question"],
            "intent": g["intent"],
            "slots": list(g["slot_sig"]),
            "required_evidence_roles": list(g["role_sig"]),
            "answer_shape": g["answer_shape"],
            "example_raw_questions": g["example_raw_questions"],
            "support_fragment_ids": sorted(g["support_fragment_ids"]),
            "support_document_ids": sorted(g["support_document_ids"]),
            "support_fragment_count": len(g["support_fragment_ids"]),
            "support_document_count": len(g["support_document_ids"]),
            "class_candidates": [],  # filled by class discovery
        })

    # build fragment-question edges
    for rec in affordances:
        for d in rec["directly_answerable"]:
            key = (d["intent"], _slot_signature(d["slots"]),
                   tuple(sorted(d["evidence_roles"])))
            pid = key_to_id.get(key)
            if not pid:
                continue
            edges.append({
                "fragment_id": rec["fragment_id"],
                "prototype_id": pid,
                "support_type": "direct",
                "confidence": d["confidence"],
                "supporting_span": d["supporting_span"],
                "evidence_roles_present": d["evidence_roles"],
            })
        for p in rec["partially_answerable"]:
            edges.append({
                "fragment_id": rec["fragment_id"],
                "prototype_id": "qproto_partial_process_overview",
                "support_type": "partial",
                "confidence": p["confidence"],
                "supporting_span": p["supporting_span"],
                "evidence_roles_present": [],
            })

    write_json(paths.out / "question_prototypes.json", prototypes)
    write_csv(paths.out / "question_prototypes.csv", prototypes, columns=[
        "prototype_id", "canonical_name", "intent", "slots",
        "required_evidence_roles", "answer_shape", "support_fragment_count",
        "support_document_count", "example_raw_questions"])
    write_csv(paths.out / "fragment_question_edges.csv", edges, columns=[
        "fragment_id", "prototype_id", "support_type", "confidence",
        "supporting_span", "evidence_roles_present"])

    log("QPROTO", f"Raw questions generated: {raw_count}")
    log("QPROTO", f"Normalized prototypes: {len(prototypes)}")
    log("QPROTO", f"Fragment-question edges: {len(edges)}")
    log("QPROTO", "Top prototypes by supporting fragments:")
    for p in prototypes[:20]:
        sub(f"{p['canonical_name']}: {p['support_fragment_count']} fragments")

    intent_counter = Counter(p["intent"] for p in prototypes)
    plotting.bar(paths.plots / "question_prototype_frequency.png",
                 [p["canonical_name"] for p in prototypes[:20]],
                 [p["support_fragment_count"] for p in prototypes[:20]],
                 "Question prototype frequency (top 20)", ylabel="fragments",
                 horizontal=True)
    plotting.bar(paths.plots / "top_intents.png",
                 [i for i, _ in intent_counter.most_common()],
                 [c for _, c in intent_counter.most_common()],
                 "Prototype count by intent", ylabel="prototypes")
    plotting.hist(paths.plots / "fragments_per_question_prototype.png",
                  [p["support_fragment_count"] for p in prototypes],
                  "Fragments per question prototype", "fragments", bins=12)
    log("QPROTO", f"Saved: {paths.out / 'question_prototypes.json'}")
    return prototypes, edges
