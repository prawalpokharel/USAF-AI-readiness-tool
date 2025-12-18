# USAF AI Readiness & Governance Assessment Tool (Demo)

A public, non-operational assessment tool that scores AI readiness and governance maturity across:
- Data & Interoperability
- Infrastructure & MLOps
- Model Governance & Assurance
- Responsible AI & Ethics
- Workforce & Change Management
- Deployment & Sustainment

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app/main.py
```

## Notes (Safety Scope)
- This demo is for governance/readiness planning only.
- Do not input classified or sensitive information.
- Not an official USAF/DoD system.
