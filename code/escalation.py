"""Multi-signal safety routing for support-ticket triage (deterministic)."""

from __future__ import annotations

from dataclasses import dataclass

from utils import normalize_text, tokenize

# Normalized retrieval confidence from retriever (0.0–1.0).
RETRIEVAL_CONFIDENCE_RISK_THRESHOLD = 0.12
# Ticket tokens overlapping retrieved chunk vs ticket token count.
CORPUS_TOKEN_OVERLAP_RISK_THRESHOLD = 0.10
# Escalate when cumulative risk score reaches this value.
RISK_ESCALATE_THRESHOLD = 4.0

SENSITIVE_KEYWORD_SIGNALS: tuple[tuple[str, str], ...] = (
    ("fraud", "fraud"),
    ("refund", "refund"),
    ("billing", "billing"),
    ("unauthorized", "unauthorized"),
    ("compromised", "compromised"),
    ("security", "security"),
    ("password_reset", "password reset"),
    ("account_locked", "account locked"),
)

# Distinct intent families for multi-intent scoring (deterministic ordering).
INTENT_BILLING = "billing"
INTENT_ACCOUNT_ACCESS = "account_access"
INTENT_SECURITY = "security"

MULTI_INTENT_PAIRS: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({INTENT_BILLING, INTENT_ACCOUNT_ACCESS}), "billing_plus_account_access"),
    (frozenset({INTENT_BILLING, INTENT_SECURITY}), "billing_plus_security"),
    (frozenset({INTENT_ACCOUNT_ACCESS, INTENT_SECURITY}), "account_access_plus_security"),
)


@dataclass(frozen=True)
class RiskAssessment:
    """Structured risk evaluation output."""

    risk_score: float
    risk_flags: list[str]
    escalate: bool
    justification: str


@dataclass(frozen=True)
class EscalationDecision:
    should_escalate: bool
    reason: str


def evaluate_risk(
    issue: str,
    subject: str,
    retrieval_confidence: float,
    detected_domains: list[str],
    intents: list[str],
    *,
    corpus_token_overlap_ratio: float | None = None,
    multi_intent_pipeline: bool = False,
) -> RiskAssessment:
    """Compute deterministic multi-signal risk score and escalation recommendation.

    Parameters
    ----------
    retrieval_confidence:
        Normalized confidence from retrieval (0.0–1.0).
    detected_domains:
        Product domains inferred for the ticket (e.g. hackerrank, claude, visa).
    intents:
        Canonical intent labels (e.g. billing, account_access).
    corpus_token_overlap_ratio:
        |ticket_tokens ∩ chunk_tokens| / |ticket_tokens| for the best retrieved
        chunk; pass ``None`` to skip the unsupported-topic signal.
    """
    text = normalize_text(f"{subject} {issue}")
    risk_score = 0.0
    flags: list[str] = []

    # 1) Sensitive keywords: +2 per match (phrases and single tokens).
    for flag_id, needle in SENSITIVE_KEYWORD_SIGNALS:
        if needle in text:
            risk_score += 2.0
            flags.append(f"sensitive_keyword:{flag_id}")

    # 2) Low retrieval confidence: +3
    if retrieval_confidence < RETRIEVAL_CONFIDENCE_RISK_THRESHOLD:
        risk_score += 3.0
        flags.append("low_retrieval_confidence")

    # 3) Cross-domain ambiguity: +2 (multiple distinct domains detected).
    unique_domains = sorted(set(d.strip().lower() for d in detected_domains if d.strip()))
    if len(unique_domains) > 1:
        risk_score += 2.0
        flags.append("cross_domain_ambiguity")

    # 4a) Pipeline multi-intent (billing/login/access/bug/etc.): +2
    if multi_intent_pipeline:
        risk_score += 2.0
        flags.append("multi_intent_pipeline")

    # 4b) Multi-intent: +3 when conflicting intent families co-occur.
    intent_set = frozenset(i.strip().lower() for i in intents if i.strip())
    for pair_set, pair_label in MULTI_INTENT_PAIRS:
        if pair_set <= intent_set:
            risk_score += 3.0
            flags.append(f"multi_intent:{pair_label}")
            break

    # 5) Unsupported topic (low overlap with retrieved corpus text): +4
    if corpus_token_overlap_ratio is not None:
        if corpus_token_overlap_ratio < CORPUS_TOKEN_OVERLAP_RISK_THRESHOLD:
            risk_score += 4.0
            flags.append("unsupported_topic_low_corpus_overlap")

    escalate = risk_score >= RISK_ESCALATE_THRESHOLD

    justification = _build_risk_justification(escalate=escalate, risk_score=risk_score, flags=flags)

    flags_sorted = sorted(flags)
    return RiskAssessment(
        risk_score=risk_score,
        risk_flags=flags_sorted,
        escalate=escalate,
        justification=justification,
    )


def _build_risk_justification(
    *,
    escalate: bool,
    risk_score: float,
    flags: list[str],
) -> str:
    """Deterministic human-readable justification (single primary reason)."""
    if not escalate:
        return f"No escalation (risk_score={risk_score:.1f} < {RISK_ESCALATE_THRESHOLD:g})."

    flag_set = set(flags)
    # Priority-ordered: billing/refund → low confidence → unsupported → cross-domain → multi-intent → sensitive → aggregate.
    if "sensitive_keyword:billing" in flag_set or "sensitive_keyword:refund" in flag_set:
        return "Escalated due to billing-related request requiring human support."
    if "low_retrieval_confidence" in flag_set:
        return "Escalated due to low corpus match confidence."
    if "unsupported_topic_low_corpus_overlap" in flag_set:
        return "Escalated due to insufficient overlap between the ticket and local support documentation."
    if "cross_domain_ambiguity" in flag_set:
        return "Escalated due to cross-domain ambiguity requiring human routing."
    if any(f.startswith("multi_intent:") for f in flags):
        return "Escalated due to multiple distinct request intents (e.g. billing and account access)."
    if any(f.startswith("sensitive_keyword:") for f in flags):
        return "Escalated due to sensitive keywords indicating fraud, security, or account safety concerns."

    return f"Escalated due to aggregated risk score {risk_score:.1f} (threshold {RISK_ESCALATE_THRESHOLD:g})."


class EscalationEngine:
    """Facade that maps ticket context + retrieval signals into an escalation decision."""

    def decide(
        self,
        issue: str,
        subject: str,
        request_type: str,
        retrieval_confidence: float,
        detected_domains: list[str],
        intents: list[str],
        corpus_token_overlap_ratio: float | None = None,
        multi_intent_pipeline: bool = False,
    ) -> EscalationDecision:
        """Return escalation decision using structured risk scoring."""
        if request_type == "invalid":
            return EscalationDecision(False, "out_of_scope_non_support")

        assessment = evaluate_risk(
            issue,
            subject,
            retrieval_confidence,
            detected_domains,
            intents,
            corpus_token_overlap_ratio=corpus_token_overlap_ratio,
            multi_intent_pipeline=multi_intent_pipeline,
        )

        if not assessment.escalate:
            return EscalationDecision(False, assessment.justification)

        return EscalationDecision(True, assessment.justification)


def corpus_token_overlap_ratio(issue: str, subject: str, chunk_text: str | None) -> float:
    """Deterministic ratio of ticket tokens that appear in chunk text."""
    q = normalize_text(f"{subject} {issue}")
    if not q.strip():
        return 0.0
    ticket_tokens = set(tokenize(q))
    if not ticket_tokens:
        return 0.0
    if not chunk_text:
        return 0.0
    chunk_tokens = set(tokenize(chunk_text))
    overlap = len(ticket_tokens & chunk_tokens)
    return overlap / max(1, len(ticket_tokens))
