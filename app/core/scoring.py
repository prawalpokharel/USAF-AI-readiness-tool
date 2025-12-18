from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from .rubric import Rubric


@dataclass(frozen=True)
class ScoreBreakdown:
    domain_scores: Dict[str, int]
    overall_score: int
    maturity_band: str


def compute_domain_scores(rubric: Rubric, responses: Dict[str, int]) -> Dict[str, int]:
    domain_scores: Dict[str, int] = {}
    for d in rubric.domains:
        n = len(d.questions)
        if n == 0:
            domain_scores[d.id] = 0
            continue
        total = 0
        for q in d.questions:
            total += int(responses.get(q.id, 0))
        domain_raw = total / (n * rubric.scale_max)
        domain_scores[d.id] = int(round(domain_raw * 100))
    return domain_scores


def compute_overall_score(rubric: Rubric, domain_scores: Dict[str, int]) -> int:
    overall = 0.0
    for d in rubric.domains:
        overall += domain_scores.get(d.id, 0) * d.weight
    return int(round(overall))


def maturity_band(overall_score: int) -> str:
    if overall_score <= 39:
        return "Early"
    if overall_score <= 59:
        return "Developing"
    if overall_score <= 79:
        return "Operational"
    return "Leading"


def score_assessment(rubric: Rubric, responses: Dict[str, int]) -> ScoreBreakdown:
    ds = compute_domain_scores(rubric, responses)
    ov = compute_overall_score(rubric, ds)
    band = maturity_band(ov)
    return ScoreBreakdown(domain_scores=ds, overall_score=ov, maturity_band=band)
