from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import json


@dataclass(frozen=True)
class Condition:
    question_id: str
    op: str
    value: Any


@dataclass(frozen=True)
class RiskRule:
    id: str
    severity: str
    when: List[Condition]
    message: str
    evidence_needed: List[str]
    recommendation_ids: List[str]


@dataclass(frozen=True)
class TriggeredRisk:
    id: str
    severity: str
    message: str
    evidence_needed: List[str]
    recommendation_ids: List[str]


def load_risk_rules(path: str) -> List[RiskRule]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rules: List[RiskRule] = []
    for r in raw.get("rules", []):
        conds = [Condition(question_id=c["question_id"], op=c["op"], value=c["value"]) for c in r.get("when", [])]
        rules.append(RiskRule(
            id=r["id"],
            severity=r["severity"],
            when=conds,
            message=r["message"],
            evidence_needed=list(r.get("evidence_needed", [])),
            recommendation_ids=list(r.get("recommendation_ids", []))
        ))
    return rules


def evaluate_condition(selected_score: int, op: str, value: Any) -> bool:
    if op == "==":
        return selected_score == int(value)
    if op == "<=":
        return selected_score <= int(value)
    if op == ">=":
        return selected_score >= int(value)
    if op == "<":
        return selected_score < int(value)
    if op == ">":
        return selected_score > int(value)
    if op == "in":
        return selected_score in value
    if op == "not_in":
        return selected_score not in value
    raise ValueError(f"Unsupported operator: {op}")


def rule_triggers(rule: RiskRule, responses: Dict[str, int]) -> bool:
    for c in rule.when:
        v = int(responses.get(c.question_id, 0))
        if not evaluate_condition(v, c.op, c.value):
            return False
    return True


def evaluate_risks(rules: List[RiskRule], responses: Dict[str, int]) -> List[TriggeredRisk]:
    out: List[TriggeredRisk] = []
    for r in rules:
        if rule_triggers(r, responses):
            out.append(TriggeredRisk(
                id=r.id,
                severity=r.severity,
                message=r.message,
                evidence_needed=r.evidence_needed,
                recommendation_ids=r.recommendation_ids
            ))
    # Sort: RED first
    out.sort(key=lambda x: (0 if x.severity == "RED" else 1, x.id))
    return out
