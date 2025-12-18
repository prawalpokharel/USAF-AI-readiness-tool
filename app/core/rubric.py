from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json


@dataclass(frozen=True)
class RubricOption:
    label: str
    score: int


@dataclass(frozen=True)
class RubricQuestion:
    id: str
    text: str
    options: List[RubricOption]


@dataclass(frozen=True)
class RubricDomain:
    id: str
    name: str
    weight: float
    questions: List[RubricQuestion]


@dataclass(frozen=True)
class Rubric:
    version: str
    title: str
    scale_min: int
    scale_max: int
    domains: List[RubricDomain]


def load_rubric(path: str) -> Rubric:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    scale = raw.get("scale", {})
    scale_min = int(scale.get("min", 0))
    scale_max = int(scale.get("max", 3))

    domains: List[RubricDomain] = []
    for d in raw.get("domains", []):
        questions: List[RubricQuestion] = []
        for q in d.get("questions", []):
            opts = [RubricOption(label=o["label"], score=int(o["score"])) for o in q.get("options", [])]
            questions.append(RubricQuestion(id=q["id"], text=q["text"], options=opts))
        domains.append(RubricDomain(
            id=d["id"],
            name=d["name"],
            weight=float(d["weight"]),
            questions=questions
        ))

    # basic validation
    wsum = sum(d.weight for d in domains)
    if abs(wsum - 1.0) > 1e-6:
        raise ValueError(f"Domain weights must sum to 1.0; got {wsum}")

    return Rubric(
        version=str(raw.get("version", "")),
        title=str(raw.get("title", "")),
        scale_min=scale_min,
        scale_max=scale_max,
        domains=domains
    )


def list_all_questions(rubric: Rubric) -> List[RubricQuestion]:
    out: List[RubricQuestion] = []
    for d in rubric.domains:
        out.extend(d.questions)
    return out


def get_domain_by_id(rubric: Rubric, domain_id: str) -> Optional[RubricDomain]:
    for d in rubric.domains:
        if d.id == domain_id:
            return d
    return None


def get_question_by_id(rubric: Rubric, question_id: str) -> Optional[RubricQuestion]:
    for d in rubric.domains:
        for q in d.questions:
            if q.id == question_id:
                return q
    return None
