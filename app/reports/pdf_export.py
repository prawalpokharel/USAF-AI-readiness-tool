from __future__ import annotations
from typing import Any, Dict, List
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


def export_pdf(path: str, report: Dict[str, Any]) -> None:
    c = canvas.Canvas(path, pagesize=letter)
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
    line(f"Generated: {report.get('generated_at','')}")
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
            line(f" - [{r['severity']}] {r['id']}: {r['message']}")

    line("")
    line("Top Priority Actions:")
    for item in report.get("top_actions", []):
        line(f" - {item['title']} (Effort: {item['effort']})")

    line("")
    line("Disclaimer: Not an official USAF/DoD product. Do not input classified or sensitive information.")
    c.save()
