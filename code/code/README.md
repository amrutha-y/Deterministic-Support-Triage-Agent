# Deterministic Support Triage Agent

This module implements a terminal-based, deterministic support triage pipeline for HackerRank Orchestrate.

## What it does

- Reads `support_tickets/support_tickets.csv`
- Classifies each ticket (`request_type`, `product_area`)
- Retrieves relevant support documentation from local `data/`
- Applies deterministic risk escalation rules (`replied` vs `escalated`)
- Generates grounded response + justification
- Writes `support_tickets/output.csv` with columns:
  - `status`
  - `product_area`
  - `response`
  - `justification`
  - `request_type`

## Setup

From repo root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional:
- Create `.env` from `.env.example` for local flags.
- Set `ENABLE_EMBEDDINGS=1` to enable sentence-transformers retrieval if dependencies/model are available locally.
- Default mode uses deterministic keyword retrieval (no external services required).

## Run

```bash
python code/main.py
```

## Determinism

- Stable corpus file ordering and chunk IDs
- Rule-based classifier and escalation policy
- Deterministic ranking tie-breakers (`score desc`, `source_path asc`, `chunk_id asc`)
- No random sampling

## Notes

- Uses only local corpus in `data/`; no web APIs for answer grounding.
- Escalates sensitive/high-risk or low-coverage cases.
- Handles blank subject/issue and `company=None`.
