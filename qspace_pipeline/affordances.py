"""Component 4: heuristic answer-affordance detection.

No external LLM/API is assumed to be available, so this module implements a
transparent, rule-based fallback. It is loudly reported as heuristic in logs
and in the final report. The design mirrors the strict-JSON LLM prompt in the
specification: every *directly answerable* question must carry an exact
supporting span copied from the fragment; otherwise it is downgraded to
*partial* or dropped.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from . import plotting
from .config import Paths
from .utils import (banner, log, split_sentences, sub, truncate, write_csv,
                    write_jsonl)

USED_LLM = False  # heuristic fallback flag, surfaced in the report

# ---------------------------------------------------------------------------
# Reusable value patterns for evidence-role extraction.
# ---------------------------------------------------------------------------
_REGION = re.compile(
    r"\b(India|United States|USA|US|Europe|EU|UK|United Kingdom|Global|"
    r"any region|all regions)\b", re.IGNORECASE)
_AMOUNT = re.compile(
    r"(?:over|above|under|up to|exceed|below|more than)?\s*"
    r"(\$\s?[\d,]+(?:\.\d+)?\s*(?:million|thousand|k)?|\b\d+\s?k\b)",
    re.IGNORECASE)
_DATE = re.compile(
    r"\b((?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2})")
_PERIOD = re.compile(r"\b(Q[1-4]\s?\d{4}|FY\s?\d{4}|full[- ]year\s?\d{4})\b",
                     re.IGNORECASE)
_PERCENT = re.compile(r"\b(\d+(?:\.\d+)?\s?(?:percent|%))")
_MONEY_VALUE = re.compile(r"(\$\s?[\d,]+(?:\.\d+)?\s*(?:million|M|thousand|k)?|"
                          r"\b\d+(?:\.\d+)?\s?(?:percent|%))", re.IGNORECASE)
_APPROVER = re.compile(
    r"\b((?:Chief [A-Z][a-z]+ Officer|C[EFIOR]O|Finance Director|Regional VP|"
    r"Finance Manager|Line Manager|Department Head|FP&A Director|Data Owner|"
    r"Chief Data Officer|Chief Risk Officer|Chief Procurement Officer|"
    r"Procurement Manager|IT Security|Legal|Data Protection Officer|"
    r"Incident Commander|CISO|IT Service Desk Manager))\b")
_OWNER = re.compile(
    r"\b(The\s+)?([A-Z][A-Za-z&]+(?:\s+[A-Z][A-Za-z&]+){0,3}?\s+"
    r"(?:team|Team|Office|Center|Operations|Desk))\b")
_SLA = re.compile(r"within\s+(\d+\s+(?:minutes|hours|days|business days))",
                  re.IGNORECASE)
_ESCALATE = re.compile(r"escalated?\s+to\s+(?:the\s+)?([A-Z][A-Za-z ]+?)"
                       r"(?:\.|,|;|$| within| and)", re.IGNORECASE)
_METRIC = re.compile(
    r"\b(revenue|gross margin|operating expenses|subscription revenue|"
    r"services revenue|headcount cost|forecast)\b", re.IGNORECASE)
_ROLE_PERSON = re.compile(
    r"\b(The\s+)?([A-Z][A-Za-z ]+?)\s+is responsible\b")
_DRIVER = re.compile(r"driven by\s+(.+?)(?:\.|;|$)", re.IGNORECASE)


def _first(pat: re.Pattern, text: str) -> Optional[str]:
    m = pat.search(text)
    if not m:
        return None
    # last non-empty group
    for g in reversed(m.groups()):
        if g and g.strip():
            return g.strip()
    return m.group(0).strip()


def _exact_span(fragment_text: str, sentence: str) -> Optional[str]:
    """Return an exact substring of ``fragment_text`` matching ``sentence``.

    Enforces the validation rule: a direct question keeps a span only if the
    span truly occurs (modulo whitespace) inside the fragment text.
    """
    if sentence in fragment_text:
        return sentence
    norm = re.sub(r"\s+", " ", sentence).strip()
    hay = re.sub(r"\s+", " ", fragment_text)
    idx = hay.find(norm)
    if idx == -1:
        return None
    # Map back to a real slice of the original text by re-searching words.
    words = norm.split()
    if not words:
        return None
    pat = re.compile(re.escape(words[0]) + r".*?" + re.escape(words[-1]),
                     re.DOTALL)
    m = pat.search(fragment_text)
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# Intent detectors. Each returns direct/partial affordance dicts for a sentence.
# ---------------------------------------------------------------------------

def _detect_approval(sent: str) -> Optional[Dict]:
    if not re.search(r"\bapprov", sent, re.IGNORECASE):
        return None
    region = _first(_REGION, sent)
    amount = _first(_AMOUNT, sent)
    approver = _first(_APPROVER, sent)
    if not approver:
        return None
    slots = {}
    roles = ["approver"]
    if region:
        slots["region"] = region
        roles.insert(0, "region")
    if amount:
        slots["amount_threshold"] = amount
        roles.insert(1 if region else 0, "amount_threshold")
    ttype = "purchase" if re.search(r"purchas", sent, re.I) else (
        "expense" if re.search(r"expens|claim|reimburs", sent, re.I) else
        "transaction")
    slots["transaction_type"] = ttype
    q = f"Who approves {ttype}s"
    if amount:
        q += f" over {amount}"
    if region:
        q += f" in {region}"
    q += "?"
    role_values = {r: slots.get(r) for r in roles if slots.get(r)}
    role_values["approver"] = approver
    return {
        "intent": "approval_lookup",
        "raw_question": q,
        "slots": slots,
        "answer_shape": ["approver", "condition", "source"],
        "evidence_roles": roles,
        "role_values": role_values,
        "confidence": 0.9,
    }


def _detect_threshold(sent: str) -> Optional[Dict]:
    amount = _first(_AMOUNT, sent)
    if not amount or not re.search(
            r"\b(over|above|under|up to|exceed|below|threshold|more than|"
            r"no longer than|not exceed)\b", sent, re.IGNORECASE):
        return None
    if re.search(r"\bapprov", sent, re.IGNORECASE):
        return None  # handled by approval detector
    region = _first(_REGION, sent)
    slots = {"amount_threshold": amount}
    roles = ["amount_threshold"]
    if region:
        slots["region"] = region
        roles.insert(0, "region")
    q = f"What is the threshold of {amount}"
    if region:
        q += f" in {region}"
    q += " and what condition applies?"
    return {
        "intent": "threshold_lookup",
        "raw_question": q,
        "slots": slots,
        "answer_shape": ["threshold", "condition", "source"],
        "evidence_roles": roles,
        "role_values": {r: slots[r] for r in roles},
        "confidence": 0.78,
    }


def _detect_owner(sent: str) -> Optional[Dict]:
    if not re.search(r"\b(owns|owner|responsible for (this|the) )", sent,
                     re.IGNORECASE):
        return None
    m = _OWNER.search(sent)
    owner = (m.group(2).strip() if m else None)
    if not owner:
        return None
    obj = "this policy" if re.search(r"policy", sent, re.I) else (
        "this SOP" if re.search(r"SOP|procedure", sent, re.I) else "this item")
    return {
        "intent": "owner_lookup",
        "raw_question": f"Who owns {obj}?",
        "slots": {"object": obj},
        "answer_shape": ["owner", "object", "source"],
        "evidence_roles": ["owner"],
        "role_values": {"owner": owner},
        "confidence": 0.8,
    }


def _detect_effective_date(sent: str) -> Optional[Dict]:
    if not re.search(r"\beffective\b", sent, re.IGNORECASE):
        return None
    date = _first(_DATE, sent)
    if not date:
        return None
    return {
        "intent": "effective_date_lookup",
        "raw_question": "When did this document become effective?",
        "slots": {"document": "this document"},
        "answer_shape": ["effective_date", "source"],
        "evidence_roles": ["effective_date"],
        "role_values": {"effective_date": date},
        "confidence": 0.85,
    }


def _detect_applicability(sent: str) -> Optional[Dict]:
    if not re.search(r"\b(applies to|scope|applicab)", sent, re.IGNORECASE):
        return None
    m = re.search(r"applies to\s+(.+?)(?:\.|;|$)", sent, re.IGNORECASE)
    audience = m.group(1).strip() if m else "the stated audience"
    return {
        "intent": "applicability_check",
        "raw_question": f"Does this apply to {truncate(audience, 60)}?",
        "slots": {"audience": truncate(audience, 60)},
        "answer_shape": ["scope", "audience", "source"],
        "evidence_roles": ["scope", "audience"],
        "role_values": {"scope": truncate(audience, 80), "audience": truncate(audience, 60)},
        "confidence": 0.7,
    }


def _detect_exception(sent: str) -> Optional[Dict]:
    if not re.search(r"\b(except|exception|unless|prohibited)\b", sent,
                     re.IGNORECASE):
        return None
    m = re.search(r"(?:except|unless)\s+(.+?)(?:\.|;|$)", sent, re.IGNORECASE)
    cond = m.group(1).strip() if m else "the stated condition"
    return {
        "intent": "exception_lookup",
        "raw_question": "What is the exception to this rule?",
        "slots": {"condition": truncate(cond, 60)},
        "answer_shape": ["exception", "condition", "source"],
        "evidence_roles": ["exception", "condition"],
        "role_values": {"exception": truncate(cond, 80), "condition": truncate(cond, 60)},
        "confidence": 0.72,
    }


def _detect_rule(sent: str) -> Optional[Dict]:
    if not re.search(r"\b(must|shall|require|required|may not|must not|"
                     r"prohibited)\b", sent, re.IGNORECASE):
        return None
    subj_m = re.match(r"\s*([A-Z][A-Za-z ]+?)\s+(?:must|shall|require|are|is)",
                      sent)
    subject = subj_m.group(1).strip() if subj_m else "the subject"
    return {
        "intent": "rule_lookup",
        "raw_question": f"What rule applies to {truncate(subject, 40)}?",
        "slots": {"subject": truncate(subject, 40)},
        "answer_shape": ["rule", "condition", "source"],
        "evidence_roles": ["rule", "condition"],
        "role_values": {"rule": truncate(sent, 90)},
        "confidence": 0.62,
    }


def _detect_metric(sent: str) -> Optional[Dict]:
    metric = _first(_METRIC, sent)
    value = _first(_MONEY_VALUE, sent)
    if not metric or not value:
        return None
    period = _first(_PERIOD, sent)
    slots = {"metric": metric}
    roles = ["metric", "value"]
    if period:
        slots["period"] = period
        roles.append("period")
    rv = {"metric": metric, "value": value}
    if period:
        rv["period"] = period
    return {
        "intent": "metric_lookup",
        "raw_question": f"What was {metric}" + (f" in {period}" if period else "") + "?",
        "slots": slots,
        "answer_shape": ["metric", "value", "period", "source"],
        "evidence_roles": roles,
        "role_values": rv,
        "confidence": 0.8,
    }


def _detect_variance(sent: str) -> Optional[Dict]:
    if not re.search(r"\b(variance|below forecast|exceeded forecast|"
                     r"driven by)\b", sent, re.IGNORECASE):
        return None
    driver = _first(_DRIVER, sent)
    pct = _first(_PERCENT, sent)
    rv = {}
    roles = ["variance"]
    if pct:
        rv["variance"] = pct
    if driver:
        rv["variance_driver"] = truncate(driver, 70)
        roles.append("variance_driver")
    if not rv:
        rv["variance"] = truncate(sent, 80)
    return {
        "intent": "variance_explanation",
        "raw_question": "What drove the variance and by how much?",
        "slots": {"metric": _first(_METRIC, sent) or "the metric"},
        "answer_shape": ["variance", "driver", "period", "source"],
        "evidence_roles": roles,
        "role_values": rv,
        "confidence": 0.7,
    }


def _detect_sla(sent: str) -> Optional[Dict]:
    sla = _first(_SLA, sent)
    if not sla and not re.search(r"\bSLA\b", sent):
        return None
    if not sla:
        m = re.search(r"SLA of\s+(\d+\s+\w+(?:\s+\w+)?)", sent, re.IGNORECASE)
        sla = m.group(1) if m else None
    if not sla:
        return None
    return {
        "intent": "sla_lookup",
        "raw_question": "What is the SLA for this step?",
        "slots": {"step": "this step"},
        "answer_shape": ["sla", "condition", "source"],
        "evidence_roles": ["sla"],
        "role_values": {"sla": sla},
        "confidence": 0.78,
    }


def _detect_next_step(sent: str) -> Optional[Dict]:
    m = re.match(r"\s*Step\s+(\d+):\s*(.+)", sent, re.IGNORECASE)
    if not m:
        return None
    num, desc = m.group(1), m.group(2).strip()
    owner_m = _ROLE_PERSON.search(sent)
    owner = owner_m.group(2).strip() if owner_m else None
    roles = ["step"]
    rv = {"step": f"Step {num}: {truncate(desc, 60)}"}
    if owner:
        roles.append("step_owner")
        rv["step_owner"] = owner
    return {
        "intent": "next_step_lookup",
        "raw_question": f"What happens at step {num} and who is responsible?",
        "slots": {"step_number": num},
        "answer_shape": ["step", "owner", "source"],
        "evidence_roles": roles,
        "role_values": rv,
        "confidence": 0.75,
    }


def _detect_escalation(sent: str) -> Optional[Dict]:
    target = _first(_ESCALATE, sent)
    if not target:
        return None
    return {
        "intent": "escalation_lookup",
        "raw_question": "Who is the escalation contact if the SLA is missed?",
        "slots": {"trigger": "SLA breach"},
        "answer_shape": ["escalation_target", "trigger", "source"],
        "evidence_roles": ["escalation_target"],
        "role_values": {"escalation_target": target},
        "confidence": 0.76,
    }


_DETECTORS = [
    _detect_approval, _detect_threshold, _detect_owner, _detect_effective_date,
    _detect_applicability, _detect_exception, _detect_rule, _detect_metric,
    _detect_variance, _detect_sla, _detect_next_step, _detect_escalation,
]

# Broad "partial" intents keyed off surface cues (process-level questions).
_PARTIAL_CUES = [
    (r"approv", "What is the full approval process end to end?",
     ["workflow steps", "exception process", "escalation path"]),
    (r"Step\s+\d+|procedure|SOP", "What is the complete procedure with all steps?",
     ["all step details", "step dependencies", "rollback steps"]),
    (r"variance|forecast|revenue", "What is the full financial explanation?",
     ["baseline", "corrective action", "owner sign-off"]),
]


def detect_affordances(paths: Paths, fragments: List[Dict]) -> List[Dict]:
    banner("AFFORDANCE", "Answer-affordance detection (heuristic fallback)")
    log("AFFORDANCE", "No LLM/API configured -> using transparent rule-based "
                      "detectors. Every direct question is span-validated.")
    records: List[Dict] = []
    direct_total = partial_total = 0
    span_rejects = 0
    intent_counter: Counter[str] = Counter()
    per_frag_counts: List[int] = []
    direct_vs_partial = {"direct": 0, "partial": 0}

    for frag in fragments:
        text = frag["text"]
        if frag["fragment_type"] == "section" or len(text) < 15:
            continue
        sentences = split_sentences(text) or [text]
        direct: List[Dict] = []
        partial: List[Dict] = []
        seen_keys = set()
        for sent in sentences:
            for det in _DETECTORS:
                res = det(sent)
                if not res:
                    continue
                span = _exact_span(text, sent)
                if span is None:
                    span_rejects += 1
                    continue  # cannot validate -> drop
                key = (res["intent"], tuple(sorted(res["slots"].items())))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                res["supporting_span"] = span
                direct.append(res)
                intent_counter[res["intent"]] += 1

        # partial affordances at the fragment level
        for cue, q, missing in _PARTIAL_CUES:
            if re.search(cue, text, re.IGNORECASE):
                span = _exact_span(text, sentences[0]) or text[:120]
                partial.append({
                    "raw_question": q,
                    "missing_evidence": missing,
                    "supporting_span": span,
                    "confidence": 0.55,
                })
                break

        if not direct and not partial:
            continue
        direct_total += len(direct)
        partial_total += len(partial)
        per_frag_counts.append(len(direct) + len(partial))
        direct_vs_partial["direct"] += len(direct)
        direct_vs_partial["partial"] += len(partial)
        records.append({
            "fragment_id": frag["fragment_id"],
            "document_id": frag["document_id"],
            "fragment_text": text,
            "directly_answerable": direct,
            "partially_answerable": partial,
            "notes": "heuristic rule-based extraction; spans validated",
        })

    write_jsonl(paths.intermediate / "answer_affordances.jsonl", records)

    # flatten to CSV
    flat: List[Dict] = []
    for r in records:
        for d in r["directly_answerable"]:
            flat.append({
                "fragment_id": r["fragment_id"],
                "document_id": r["document_id"],
                "support_type": "direct",
                "intent": d["intent"],
                "raw_question": d["raw_question"],
                "slots": d["slots"],
                "evidence_roles": d["evidence_roles"],
                "supporting_span": d["supporting_span"],
                "confidence": d["confidence"],
            })
        for p in r["partially_answerable"]:
            flat.append({
                "fragment_id": r["fragment_id"],
                "document_id": r["document_id"],
                "support_type": "partial",
                "intent": "process_overview",
                "raw_question": p["raw_question"],
                "slots": {},
                "evidence_roles": [],
                "supporting_span": p["supporting_span"],
                "confidence": p["confidence"],
            })
    write_csv(paths.out / "answer_affordances.csv", flat)

    log("AFFORDANCE", f"Fragments with affordances: {len(records)}")
    log("AFFORDANCE", f"Direct questions: {direct_total} | "
                      f"Partial questions: {partial_total}")
    log("AFFORDANCE", f"Span-validation rejects: {span_rejects}")
    log("AFFORDANCE", "Top intents:")
    for intent, c in intent_counter.most_common(8):
        sub(f"{intent}: {c}")
    # example
    for r in records:
        if r["directly_answerable"]:
            d = r["directly_answerable"][0]
            log("AFFORDANCE", f"Fragment {r['fragment_id']}")
            sub(f"Text: \"{truncate(r['fragment_text'], 90)}\"")
            sub(f"- {d['raw_question']}")
            sub(f"  intent={d['intent']} "
                f"evidence_roles={d['evidence_roles']}")
            break

    plotting.hist(paths.plots / "answerable_questions_per_fragment.png",
                  per_frag_counts, "Answerable questions per fragment",
                  "questions", bins=12)
    plotting.bar(paths.plots / "intent_frequency_raw.png",
                 [i for i, _ in intent_counter.most_common()],
                 [c for _, c in intent_counter.most_common()],
                 "Raw intent frequency", ylabel="questions")
    plotting.bar(paths.plots / "direct_vs_partial_affordances.png",
                 list(direct_vs_partial.keys()),
                 list(direct_vs_partial.values()),
                 "Direct vs partial affordances", ylabel="questions",
                 color="#C44E52")
    log("AFFORDANCE", f"Saved: {paths.intermediate / 'answer_affordances.jsonl'}")
    return records
