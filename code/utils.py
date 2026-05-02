"""Utility helpers and shared data contracts for support triage."""

from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from dotenv import load_dotenv as _load_dotenv
except Exception:  # pragma: no cover - optional dependency
    _load_dotenv = None

ALLOWED_STATUS = {"replied", "escalated"}
ALLOWED_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}
OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]


@dataclass(frozen=True)
class TicketInput:
    """Input ticket record from CSV."""

    issue: str
    subject: str
    company: str
    ticket_id: int


@dataclass(frozen=True)
class CorpusChunk:
    """Corpus chunk with source metadata."""

    chunk_id: str
    text: str
    source_path: str
    company: str
    product_hints: tuple[str, ...]
    heading: str
    breadcrumbs: tuple[str, ...] = ()
    article_title: str = ""


@dataclass(frozen=True)
class RetrievalResult:
    """Result for one retrieved chunk."""

    chunk: CorpusChunk
    score: float
    method: str
    snippet: str = ""
    explanation: str = ""


@dataclass(frozen=True)
class Prediction:
    """Final output record."""

    status: str
    product_area: str
    response: str
    justification: str
    request_type: str


def load_env() -> None:
    """Load environment variables from .env when python-dotenv is available."""
    if _load_dotenv is not None:
        _load_dotenv()


def normalize_text(value: str) -> str:
    """Normalize free-form text for deterministic matching."""
    lowered = (value or "").strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def tokenize(value: str) -> list[str]:
    """Tokenize text into alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", normalize_text(value))


def join_nonempty(parts: Iterable[str], sep: str = " ") -> str:
    """Join non-empty strings with deterministic spacing."""
    return sep.join(part.strip() for part in parts if part and part.strip())


def stable_chunk_id(source_path: str, idx: int) -> str:
    """Create deterministic stable ID for chunk identity."""
    digest = hashlib.sha1(f"{source_path}::{idx}".encode("utf-8")).hexdigest()[:12]
    return f"chunk_{digest}"


def read_ticket_csv(path: Path) -> list[TicketInput]:
    """Read input support ticket CSV."""
    records: list[TicketInput] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            records.append(
                TicketInput(
                    issue=(row.get("Issue") or row.get("issue") or "").strip(),
                    subject=(row.get("Subject") or row.get("subject") or "").strip(),
                    company=(row.get("Company") or row.get("company") or "").strip(),
                    ticket_id=idx,
                )
            )
    return records


def validate_prediction(prediction: Prediction) -> None:
    """Validate output contract for one prediction."""
    if prediction.status not in ALLOWED_STATUS:
        raise ValueError(f"Invalid status: {prediction.status}")
    if prediction.request_type not in ALLOWED_REQUEST_TYPES:
        raise ValueError(f"Invalid request_type: {prediction.request_type}")
    if not prediction.product_area:
        raise ValueError("product_area cannot be empty")
    if not prediction.response:
        raise ValueError("response cannot be empty")
    if not prediction.justification:
        raise ValueError("justification cannot be empty")


def write_output_csv(path: Path, predictions: list[Prediction]) -> None:
    """Write final predictions in exact expected schema order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for prediction in predictions:
            validate_prediction(prediction)
            writer.writerow(
                {
                    "status": prediction.status,
                    "product_area": prediction.product_area,
                    "response": prediction.response,
                    "justification": prediction.justification,
                    "request_type": prediction.request_type,
                }
            )


def repo_root_from_code_dir(code_file: str) -> Path:
    """Resolve repo root path from any code module file."""
    return Path(code_file).resolve().parents[1]


def company_slug(value: str) -> str:
    """Normalize company naming to expected slugs."""
    normalized = normalize_text(value)
    if normalized in {"hackerrank", "claude", "visa"}:
        return normalized
    return "none"


def getenv_flag(name: str, default: bool = False) -> bool:
    """Read bool-like env flags deterministically."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return normalize_text(raw) in {"1", "true", "yes", "on"}
