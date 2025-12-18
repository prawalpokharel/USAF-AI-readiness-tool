# app.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


# ----------------------------
# Paths / Config
# ----------------------------
ROOT = Path(__file__).resolve().parent  # this is /app
DATA_DIR = ROOT.parent / "data"            # this is project-root /data

RUBRIC_PATH = DATA_DIR / "rubric_v1.json"
RISK_PATH = DATA_DIR / "risk_rules_v1.json"
REC_PATH = DATA_DIR / "recommendations_v1.json"
MAP_PATH = DATA_DIR / "rec_mappings_v1.json"


# ----------------------------
# Data structures
# ----------------------------
@dataclass(frozen=True)
class TriggeredRisk:
    id: str
    severity: str  # "RED" or "AMBER"
    message: str
    evidence_needed: List[str]
    recommendation_ids: List[str]


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


# ----------------------------
# Loaders
# ----------------------------
def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rubric() -> Dict[str, Any]:
    rubric = load_json(RUBRIC_PATH)
    # basic validation
    domains = rubric.get("domains", [])
    wsum = sum(float(d.get("weight", 0)) for d in domains)
    if abs(wsum - 1.0) > 1e-6:
        raise ValueError(f"Rubric domain weights must sum to 1.0 (got {wsum})")
    return rubric


def load_risk_rules() -> List[Dict[str, Any]]:
    raw = load_json(RISK_PATH)
    return list(raw.get("rules", []))


def load_recommendations() -> Dict[str, Recommendation]:
    raw = load_json(REC_PATH)
    out: Dict[str, Recommendation] = {}
    for r in raw.get("recommendations", []):
        rec = Recommendation(
            id=r["id"],
            title=r["title"],
            why=r["why"],
            effort=r["effort"],
            owner_role=r["owner_role"],
            evidence_artifacts=list(r.get("evidence_artifacts", [])),
            tags=list(r.get("tags", [])),
        )
        out[rec.id] = rec
    return out


def load_rec_mappings() -> Dict[str, Dict[str, Any]]:
    raw = load_json(MAP_PATH)
    out: Dict[str, Dict[str, Any]] = {}
    for m in raw.get("question_score_triggers", []):
        out[m["question_id"]] = {
            "when_score_lte": int(m["when_score_lte"]),
            "recommendation_ids": list(m.get("recommendation_ids", [])),
        }
    return out


# ----------------------------
# Scoring
# ----------------------------
def compute_domain_scores(rubric: Dict[str, Any], responses: Dict[str, int]) -> Dict[str, int]:
    scale_max = int(rubric.get("scale", {}).get("max", 3))
    domain_scores: Dict[str, int] = {}

    for d in rubric.get("domains", []):
        qs = d.get("questions", [])
        if not qs:
            domain_scores[d["id"]] = 0
            continue
        total = sum(int(responses.get(q["id"], 0)) for q in qs)
        raw = total / (len(qs) * scale_max)
        domain_scores[d["id"]] = int(round(raw * 100))

    return domain_scores


def compute_overall_score(rubric: Dict[str, Any], domain_scores: Dict[str, int]) -> int:
    overall = 0.0
    for d in rubric.get("domains", []):
        overall += float(d["weight"]) * float(domain_scores.get(d["id"], 0))
    return int(round(overall))


def maturity_band(score: int) -> str:
    if score <= 39:
        return "Early"
    if score <= 59:
        return "Developing"
    if score <= 79:
        return "Operational"
    return "Leading"


# ----------------------------
# Risk evaluation
# ----------------------------
def eval_condition(selected_score: int, op: str, value: Any) -> bool:
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


def evaluate_risks(rules: List[Dict[str, Any]], responses: Dict[str, int]) -> List[TriggeredRisk]:
    out: List[TriggeredRisk] = []
    for r in rules:
        conds = r.get("when", [])
        ok = True
        for c in conds:
            qid = c["question_id"]
            op = c["op"]
            val = c["value"]
            score = int(responses.get(qid, 0))
            if not eval_condition(score, op, val):
                ok = False
                break
        if ok:
            out.append(
                TriggeredRisk(
                    id=r["id"],
                    severity=r["severity"],
                    message=r["message"],
                    evidence_needed=list(r.get("evidence_needed", [])),
                    recommendation_ids=list(r.get("recommendation_ids", [])),
                )
            )
    out.sort(key=lambda x: (0 if x.severity == "RED" else 1, x.id))
    return out


# ----------------------------
# Human-friendly messages
# ----------------------------
def risk_title(severity: str) -> str:
    return "Critical Risk Identified" if severity == "RED" else "Potential Risk Identified"


def human_risk_message(risk: TriggeredRisk) -> str:
    # Friendly versions for your known rules
    if risk.id == "RR_SUSTAINMENT_RISK":
        return (
            "Long-term ownership or funding is not clearly defined. "
            "When responsibility and sustainment budget aren’t assigned, AI efforts often stall after a pilot "
            "and fail to deliver sustained value over time."
        )
    if risk.id == "RR_MONITORING_NONE":
        return (
            "There is no monitoring in place to detect performance degradation or data drift after deployment. "
            "Without monitoring, issues can go unnoticed until they cause operational or compliance problems."
        )
    if risk.id == "RR_DATA_OWNERSHIP_NONE":
        return (
            "Data ownership is unclear. Without a named data owner/steward and clear governance rules, "
            "access control, accountability, and data quality typically degrade over time."
        )
    if risk.id == "RR_CHANGE_CONTROL_NONE":
        return (
            "Model changes are not governed. Without approvals, versioning, and traceability, "
            "unreviewed updates can introduce reliability and compliance risks."
        )
    if risk.id == "RR_AUDITABILITY_NONE":
        return (
            "Outputs are not auditable. Without traceable logs and reproducibility, it becomes difficult to "
            "investigate errors, prove compliance, or hold the right parties accountable."
        )
    if risk.id == "RR_PRIVACY_REVIEW_NONE":
        return (
            "No privacy review has been documented. This increases the risk of inappropriate data handling "
            "and potential compliance issues."
        )
    if risk.id == "RR_VALIDATION_WEAK":
        return (
            "Validation and testing appear limited. When models are not tested against clear acceptance criteria "
            "and realistic stress cases, failures in real-world conditions become more likely."
        )
    if risk.id == "RR_INCIDENT_RESPONSE_WEAK":
        return (
            "Incident response for model failures appears ad hoc. Without a clear runbook and rollback process, "
            "teams usually recover more slowly when problems occur."
        )
    if risk.id == "RR_HUMAN_OVERSIGHT_WEAK":
        return (
            "Human oversight is unclear. When override procedures and accountability are not defined, "
            "AI-assisted decisions can create safety and governance gaps."
        )
    if risk.id == "RR_BIAS_FAIRNESS_WEAK":
        return (
            "Bias/fairness assessment appears insufficient for the use case. Without evaluation and monitoring, "
            "unintended disparate impacts may go undetected."
        )

    # Fallback: still readable (no ID)
    return risk.message


# ----------------------------
# Recommendations
# ----------------------------
def generate_recommendations(
    responses: Dict[str, int],
    triggered_risks: List[TriggeredRisk],
    rec_library: Dict[str, Recommendation],
    rec_mappings: Dict[str, Dict[str, Any]],
) -> List[RecommendedItem]:
    items: List[RecommendedItem] = []
    seen: set[str] = set()

    # From triggered risks first
    for risk in triggered_risks:
        for rid in risk.recommendation_ids:
            if rid in rec_library and rid not in seen:
                seen.add(rid)
                items.append(RecommendedItem(rec_library[rid], reason=f"Triggered by a risk flag ({risk.severity})."))

    # From low-score questions
    for qid, score in responses.items():
        mapping = rec_mappings.get(qid)
        if not mapping:
            continue
        if score <= int(mapping["when_score_lte"]):
            for rid in mapping["recommendation_ids"]:
                if rid in rec_library and rid not in seen:
                    seen.add(rid)
                    items.append(RecommendedItem(rec_library[rid], reason=f"Recommended due to a low readiness score in this area."))

    return items


def domain_weight_for_question(rubric: Dict[str, Any], question_id: str) -> float:
    for d in rubric.get("domains", []):
        for q in d.get("questions", []):
            if q["id"] == question_id:
                return float(d.get("weight", 0.0))
    return 0.0


def pick_top_actions(
    rubric: Dict[str, Any],
    responses: Dict[str, int],
    triggered_risks: List[TriggeredRisk],
    recs: List[RecommendedItem],
    k: int = 5,
) -> List[RecommendedItem]:
    # build severity map (rec id -> best severity rank)
    red: set[str] = set()
    amber: set[str] = set()
    for r in triggered_risks:
        for rid in r.recommendation_ids:
            if r.severity == "RED":
                red.add(rid)
            else:
                amber.add(rid)

    def sev_rank(rec_id: str) -> int:
        if rec_id in red:
            return 0
        if rec_id in amber:
            return 1
        return 2

    ranked = sorted(
        recs,
        key=lambda item: (
            sev_rank(item.recommendation.id),
            -float(domain_weight_for_question(rubric, guess_question_for_item(item, responses))),
            guess_score_for_item(item, responses),
            item.recommendation.id,
        ),
    )
    return ranked[:k]


def guess_question_for_item(item: RecommendedItem, responses: Dict[str, int]) -> str:
    # We keep it simple in a single-file app: just return any key so domain weight ranking still works
    # (In a multi-file version, we’d store exact triggers per item.)
    return next(iter(responses.keys()), "")


def guess_score_for_item(item: RecommendedItem, responses: Dict[str, int]) -> int:
    # Use min score across all responses as conservative urgency signal
    if not responses:
        return 3
    return min(int(v) for v in responses.values())


# ----------------------------
# Charts
# ----------------------------
def radar_chart(labels: List[str], values: List[int]):
    angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
    angles += angles[:1]
    vals = values + values[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids([a * 180 / math.pi for a in angles[:-1]], labels)
    ax.set_ylim(0, 100)
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.2)
    ax.grid(True)
    return fig


# ----------------------------
# PDF Export
# ----------------------------
def export_pdf(path: Path, report: Dict[str, Any]) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    x = 0.75 * inch
    y = height - 0.75 * inch

    def line(txt: str, dy: float = 14):
        nonlocal y
        c.drawString(x, y, txt[:120])
        y -= dy
        if y < 0.75 * inch:
            c.showPage()
            y = height - 0.75 * inch

    line("USAF AI Readiness & Governance Assessment (Public, Non-Operational)")
    line(f"Generated: {report.get('generated_at', '')}")
    line("")
    line(f"Overall Score: {report.get('overall_score','')}  |  Band: {report.get('maturity_band','')}")
    line("")
    line("Domain Scores:")
    for name, score in report.get("domain_scores_named", []):
        line(f" - {name}: {score}")

    line("")
    line("Risk Flags:")
    risks = report.get("risks", [])
    if not risks:
        line(" - None triggered")
    else:
        for r in risks:
            line(f" - {r['severity']}: {r['message']}")

    line("")
    line("Top Priority Actions:")
    for item in report.get("top_actions", []):
        line(f" - {item['title']} (Effort: {item['effort']})")

    line("")
    line("Disclaimer: Not an official USAF/DoD product. Do not input classified or sensitive information.")
    c.save()


# ----------------------------
# UI Helpers
# ----------------------------
def init_state():
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    if "context" not in st.session_state:
        st.session_state.context = {
            "org_type": "Unit",
            "mission_area": "Logistics",
            "ai_type": "Decision Support (non-operational)",
        }


def render_question(q: Dict[str, Any], responses: Dict[str, int]):
    qid = q["id"]
    options = q.get("options", [])
    labels = [o["label"] for o in options]
    scores = [int(o["score"]) for o in options]
    default_score = int(responses.get(qid, 0))
    default_index = scores.index(default_score) if default_score in scores else 0

    choice = st.radio(q["text"], labels, index=default_index, key=f"q_{qid}")
    responses[qid] = int(scores[labels.index(choice)])


def build_report(
    rubric: Dict[str, Any],
    responses: Dict[str, int],
    domain_scores: Dict[str, int],
    overall: int,
    band: str,
    risks: List[TriggeredRisk],
    recs: List[RecommendedItem],
    top_actions: List[RecommendedItem],
) -> Dict[str, Any]:
    domain_scores_named = []
    for d in rubric.get("domains", []):
        domain_scores_named.append((d["name"], int(domain_scores.get(d["id"], 0))))

    return {
        "generated_at": st.session_state.get("generated_at", ""),
        "context": st.session_state.context,
        "responses": responses,
        "overall_score": overall,
        "maturity_band": band,
        "domain_scores": domain_scores,
        "domain_scores_named": domain_scores_named,
        "risks": [
            {
                "severity": r.severity,
                "message": human_risk_message(r),
                "evidence_needed": r.evidence_needed,
            }
            for r in risks
        ],
        "recommendations": [
            {
                "id": it.recommendation.id,
                "title": it.recommendation.title,
                "effort": it.recommendation.effort,
                "owner_role": it.recommendation.owner_role,
                "why": it.recommendation.why,
                "evidence_artifacts": it.recommendation.evidence_artifacts,
                "tags": it.recommendation.tags,
                "reason": it.reason,
            }
            for it in recs
        ],
        "top_actions": [
            {
                "id": it.recommendation.id,
                "title": it.recommendation.title,
                "effort": it.recommendation.effort,
                "owner_role": it.recommendation.owner_role,
                "reason": it.reason,
            }
            for it in top_actions
        ],
    }


# ----------------------------
# Main app
# ----------------------------
def main():
    st.set_page_config(page_title="AI Readiness & Governance Tool", layout="wide")
    init_state()

    st.title("USAF AI Readiness & Governance Assessment (Demo)")
    st.caption("Public, non-operational governance/readiness tool. Do not input classified or sensitive information.")
    # --- Front-page: What this app does + when to use it ---
    with st.expander("What this app does, when to use it, and how to use it (Read first)", expanded=True):
        st.markdown(
        """
### What this app does
This tool helps you **self-assess AI readiness and governance maturity** using a structured rubric (0–3) across six domains:
- **Data & Interoperability**
- **Infrastructure & MLOps**
- **Model Governance & Assurance**
- **Responsible AI & Ethics**
- **Workforce & Change Management**
- **Deployment & Sustainment**

It produces:
- An **overall readiness score** (0–100) and a **maturity band**
- **Risk flags** written in plain English (critical vs. potential gaps)
- A prioritized list of **recommended next actions**
- Exportable **JSON + PDF reports** for documentation and evidence

### When you should use this app
Use this app when you need to:
- Decide if an AI project is **ready for pilot or production**
- Identify **governance gaps** before deployment (monitoring, approvals, privacy, auditability, etc.)
- Create a **repeatable readiness review** for teams, programs, or research prototypes
- Generate a **professional report** you can attach to a proposal, portfolio, or compliance package

### When you should NOT use this app
Do not use this tool to:
- Make operational/tactical decisions
- Provide targeting guidance
- Handle classified or sensitive mission details

### How to use this app (recommended workflow)
1. Go to the **Assessment** tab and answer each question honestly (0–3).
2. Open the **Results** tab to review:
   - Overall score + domain scores
   - Risk flags (plain English explanations)
   - Top priority actions and recommended improvements
3. Use **Export** to download a PDF/JSON report for records or sharing.

### Safety & scope
This is a **public, non-operational demo** for governance/readiness planning only.
**Do not enter classified, sensitive, or personal data.**
        """
    )


    with st.expander("Safety scope & disclaimer", expanded=False):
        st.write(
            "- Governance/readiness planning only\n"
            "- Not official USAF/DoD policy\n"
            "- Do not input sensitive/classified data\n"
            "- No operational military advice\n"
        )

    # Load configs
    rubric = load_rubric()
    risk_rules = load_risk_rules()
    rec_library = load_recommendations()
    rec_mappings = load_rec_mappings()

    # Sidebar context
    st.sidebar.header("Assessment Context")
    st.session_state.context["org_type"] = st.sidebar.selectbox(
        "Organization type",
        ["Unit", "Program Office", "Contractor", "Academic/Research"],
        index=["Unit", "Program Office", "Contractor", "Academic/Research"].index(st.session_state.context["org_type"]),
    )
    st.session_state.context["mission_area"] = st.sidebar.selectbox(
        "Mission area (non-operational)",
        ["Logistics", "Maintenance", "Training/Education", "Cybersecurity", "Admin/HR", "Other"],
        index=["Logistics", "Maintenance", "Training/Education", "Cybersecurity", "Admin/HR", "Other"].index(
            st.session_state.context["mission_area"]
        ),
    )
    st.session_state.context["ai_type"] = st.sidebar.selectbox(
        "AI type",
        ["Analytics", "NLP", "Computer Vision (non-targeting)", "Decision Support (non-operational)", "Other"],
        index=["Analytics", "NLP", "Computer Vision (non-targeting)", "Decision Support (non-operational)", "Other"].index(
            st.session_state.context["ai_type"]
        ),
    )

    tab1, tab2, tab3 = st.tabs(["Assessment", "Results", "Methodology"])

    # ---- Assessment ----
    with tab1:
        st.subheader("Assessment Wizard")
        st.write("Scores are 0–3 (Not in place → Measured/optimized).")

        responses: Dict[str, int] = st.session_state.responses

        for domain in rubric.get("domains", []):
            with st.expander(domain["name"], expanded=(domain["id"] == "data_interop")):
                for q in domain.get("questions", []):
                    render_question(q, responses)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset answers"):
                st.session_state.responses = {}
                st.rerun()
        with col2:
            st.caption("After answering, open **Results** to view scores, risks, and recommended actions.")

    # ---- Results ----
    with tab2:
        st.subheader("Results Dashboard")

        responses = st.session_state.responses
        domain_scores = compute_domain_scores(rubric, responses)
        overall = compute_overall_score(rubric, domain_scores)
        band = maturity_band(overall)

        risks = evaluate_risks(risk_rules, responses)
        recs = generate_recommendations(responses, risks, rec_library, rec_mappings)
        top_actions = pick_top_actions(rubric, responses, risks, recs, k=5)

        # timestamp once
        if "generated_at" not in st.session_state:
            from datetime import datetime
            st.session_state.generated_at = datetime.now().isoformat(timespec="seconds")

        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Score", overall)
        c2.metric("Maturity Band", band)
        c3.metric("Risk Flags", len(risks))

        left, right = st.columns([1, 1])
        with left:
            st.markdown("### Domain Scores")
            for d in rubric.get("domains", []):
                val = int(domain_scores.get(d["id"], 0))
                st.progress(val / 100.0, text=f"{d['name']}: {val}")

        with right:
            labels = [d["name"] for d in rubric.get("domains", [])]
            vals = [int(domain_scores.get(d["id"], 0)) for d in rubric.get("domains", [])]
            if labels:
                fig = radar_chart(labels, vals)
                st.pyplot(fig)

        st.markdown("### Risk Flags (Plain English)")
        if not risks:
            st.success("No risk flags were triggered based on your inputs.")
        else:
            for r in risks:
                msg = human_risk_message(r)
                if r.severity == "RED":
                    st.error(f"🚨 {risk_title(r.severity)}\n\n{msg}")
                else:
                    st.warning(f"⚠️ {risk_title(r.severity)}\n\n{msg}")

                with st.expander("What to prepare as evidence"):
                    for e in r.evidence_needed:
                        st.write(f"- {e}")

        st.markdown("### Top Priority Actions (Executive-friendly)")
        if not top_actions:
            st.info("No priority actions identified yet. Fill out more assessment questions.")
        else:
            for item in top_actions:
                st.write(
                    f"**{item.recommendation.title}** · Effort: {item.recommendation.effort} · "
                    f"Owner: {item.recommendation.owner_role}"
                )
                st.caption(item.reason)

        st.markdown("### Recommendations (Details)")
        if not recs:
            st.info("No recommendations triggered yet.")
        else:
            for item in recs:
                with st.expander(f"{item.recommendation.title} (Effort: {item.recommendation.effort})"):
                    st.write(item.recommendation.why)
                    st.write(f"**Owner role:** {item.recommendation.owner_role}")
                    st.write("**Evidence artifacts:**")
                    for a in item.recommendation.evidence_artifacts:
                        st.write(f"- {a}")
                    st.caption(item.reason)

        st.divider()
        st.markdown("### Export")
        report = build_report(rubric, responses, domain_scores, overall, band, risks, recs, top_actions)
        report_json = json.dumps(report, indent=2)

        st.download_button(
            "Download JSON report",
            data=report_json,
            file_name="ai_readiness_report.json",
            mime="application/json",
        )

        pdf_path = ROOT / "ai_readiness_report.pdf"
        export_pdf(pdf_path, report)
        st.download_button(
            "Download PDF report",
            data=pdf_path.read_bytes(),
            file_name="ai_readiness_report.pdf",
            mime="application/pdf",
        )

    # ---- Methodology ----
    with tab3:
        st.subheader("Methodology (Public, Non-Operational)")
        st.write(
            "This tool scores readiness using a simple 0–3 rubric across six domains. "
            "Domain scores are normalized to 0–100 and the overall score is a weighted average."
        )
        st.code(
            "Domain score = (sum(question_scores) / (num_questions * 3)) * 100\n"
            "Overall score = sum(domain_score * domain_weight)",
            language="text",
        )
        st.write(
            "Risk flags are rule-based checks that identify common governance gaps. "
            "User-facing messages are presented in plain English."
        )


if __name__ == "__main__":
    main()
