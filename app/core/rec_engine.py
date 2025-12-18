from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import json

from .rubric import Rubric
from .risk_rules import TriggeredRisk


@dataclass(frozen=True)
class Recommendation:
    id: str
    title: str
    why: str
    effort: str
    owner_role: str
    evidence_artifacts: List[str]
    tags: List[str]


@dataclass(frozen=True)
class RecommendedItem:
    recommendation: Recommendation
    reason: str


def load_recommendations(path: str) -> Dict[str, Recommendation]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, Recommendation] = {}
    for r in raw.get("recommendations", []):
        rec = Recommendation(
            id=r["id"],
            title=r["title"],
            why=r["why"],
            effort=r["effort"],
            owner_role=r["owner_role"],
            evidence_artifacts=list(r.get("evidence_artifacts", [])),
            tags=list(r.get("tags", []))
        )
        out[rec.id] = rec
    return out


def load_rec_mappings(path: str) -> Dict[str, Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, Dict[str, object]] = {}
    for m in raw.get("question_score_triggers", []):
        out[m["question_id"]] = {
            "when_score_lte": int(m["when_score_lte"]),
            "recommendation_ids": list(m.get("recommendation_ids", []))
        }
    return out


def generate_recommendations(
    rubric: Rubric,
    responses: Dict[str, int],
    triggered_risks: List[TriggeredRisk],
    rec_library: Dict[str, Recommendation],
    rec_mappings: Dict[str, Dict[str, object]]
) -> List[RecommendedItem]:
    items: List[RecommendedItem] = []
    seen: Set[str] = set()

    # From risks
    for risk in triggered_risks:
        for rid in risk.recommendation_ids:
            if rid in rec_library and rid not in seen:
                seen.add(rid)
                items.append(RecommendedItem(rec_library[rid], reason=f"Triggered by {risk.id} ({risk.severity})"))

    # From low scores
    for qid, score in responses.items():
        mapping = rec_mappings.get(qid)
        if not mapping:
            continue
        if score <= int(mapping["when_score_lte"]):
            for rid in mapping["recommendation_ids"]:
                if rid in rec_library and rid not in seen:
                    seen.add(rid)
                    items.append(RecommendedItem(rec_library[rid], reason=f"Low score on {qid} (score={score})"))

    return items


def _domain_weight_for_question(rubric: Rubric, question_id: str) -> float:
    for d in rubric.domains:
        for q in d.questions:
            if q.id == question_id:
                return d.weight
    return 0.0


def pick_top_priority_actions(
    rubric: Rubric,
    responses: Dict[str, int],
    triggered_risks: List[TriggeredRisk],
    recommended: List[RecommendedItem],
    k: int = 5
) -> List[RecommendedItem]:
    # Build helper maps
    risk_rec_ids: Set[str] = set()
    red_ids: Set[str] = set()
    amber_ids: Set[str] = set()
    for r in triggered_risks:
        for rid in r.recommendation_ids:
            risk_rec_ids.add(rid)
            if r.severity == "RED":
                red_ids.add(rid)
            else:
                amber_ids.add(rid)

    def severity_rank(rec_id: str) -> int:
        if rec_id in red_ids:
            return 0
        if rec_id in amber_ids:
            return 1
        return 2

    # Tie to lowest scoring question if available
    # We'll approximate by scanning reasons
    def best_question_score(item: RecommendedItem) -> Tuple[int, float]:
        # Lower score is more urgent; domain weight higher is more urgent
        m = None
        # try extract qid from "Low score on <qid>"
        import re
        qid = None
        m = re.search(r"Low score on ([a-zA-Z0-9_\-]+)", item.reason)
        if m:
            qid = m.group(1)
        if qid and qid in responses:
            score = int(responses[qid])
            w = _domain_weight_for_question(rubric, qid)
            return (score, -w)
        return (3, -0.0)

    ranked = sorted(
        recommended,
        key=lambda item: (
            severity_rank(item.recommendation.id),
            best_question_score(item)[1],   # -domain weight
            best_question_score(item)[0],   # question score
            item.recommendation.id
        )
    )
    return ranked[:k]
