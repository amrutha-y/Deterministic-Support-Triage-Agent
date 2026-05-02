"""Top-level deterministic support triage agent."""

from __future__ import annotations

import re
from pathlib import Path

from classifier import TicketClassifier
from escalation import corpus_token_overlap_ratio, evaluate_risk
from retriever import DeterministicRetriever, RetrievalOutput
from utils import Prediction, RetrievalResult, TicketInput, join_nonempty, normalize_text, tokenize

INVALID_REPLY = "This request appears outside supported support topics."
OUT_OF_SCOPE_REPLY = "I am sorry, this is out of scope from my capabilities."
MIN_ISSUE_CHAR_THRESHOLD = 10
MIN_TICKET_TOKEN_THRESHOLD = 2
INVALID_CORPUS_OVERLAP_MAX = 0.06
INVALID_RETRIEVAL_CONFIDENCE_MAX = 0.10


class SupportTriageAgent:
    """Pipeline agent: domain detection → classification → retrieval → risk → response."""

    def __init__(self, retriever: DeterministicRetriever) -> None:
        self.retriever = retriever
        self.classifier = TicketClassifier()

    def process_ticket(self, ticket: TicketInput) -> Prediction:
        # 1) Domain detection
        inferred_company = self.classifier.infer_company(ticket.issue, ticket.subject, ticket.company)
        detected_domains = self.classifier.detect_domains(ticket.issue, ticket.subject, ticket.company)

        # Early invalid (short issue / too few tokens)
        if self._is_invalid_ticket_early(ticket):
            trace = self._decision_trace(
                detected_company=inferred_company,
                retrieval_output=None,
                pipeline_stage="skipped",
                risk_flags=["invalid_ticket_early"],
                multi_intent=False,
                final_status="replied",
            )
            return Prediction(
                status="replied",
                product_area="general_support",
                response=INVALID_REPLY,
                justification=self._justification_from_trace(
                    trace,
                    retrieval=None,
                    request_type="invalid",
                    product_area="general_support",
                    escalation_reason="ticket_below_minimum_content_threshold",
                ),
                request_type="invalid",
            )

        # 2) Request type classification
        request_type = self.classifier.classify_request_type(ticket.issue, ticket.subject)

        # Keyword-based invalid (e.g. off-topic): still retrieve for product_area metadata
        if request_type == "invalid":
            retrieval_output, pipeline_stage = self.retriever.retrieve_with_fallback_pipeline(
                ticket.issue,
                ticket.subject,
                inferred_company,
                top_k=1,
            )
            retrieval = retrieval_output.results[0] if retrieval_output.results else None
            product_area = self.classifier.classify_product_area(ticket.issue, ticket.subject, retrieval)
            trace = self._decision_trace(
                detected_company=inferred_company,
                retrieval_output=retrieval_output,
                pipeline_stage=pipeline_stage,
                risk_flags=["invalid_keyword_pattern"],
                multi_intent=False,
                final_status="replied",
            )
            return Prediction(
                status="replied",
                product_area=product_area,
                response=OUT_OF_SCOPE_REPLY,
                justification=self._justification_from_trace(
                    trace,
                    retrieval=retrieval,
                    request_type="invalid",
                    product_area=product_area,
                    escalation_reason="out_of_scope_keyword_classification",
                ),
                request_type="invalid",
            )

        # Multi-intent pipeline classes (always computed for trace + risk)
        pipeline_intents = self.classifier.detect_pipeline_intent_classes(ticket.issue, ticket.subject)
        multi_intent = len(pipeline_intents) > 1

        # 3) Retrieval (domain → global fallback)
        retrieval_output, pipeline_stage = self.retriever.retrieve_with_fallback_pipeline(
            ticket.issue,
            ticket.subject,
            inferred_company,
            top_k=1,
        )
        assert isinstance(retrieval_output, RetrievalOutput)
        retrieval = retrieval_output.results[0] if retrieval_output.results else None

        # 4) Product area (ticket keywords, else chunk metadata)
        product_area = self.classifier.classify_product_area(ticket.issue, ticket.subject, retrieval)

        # Post-retrieval invalid: very low support match
        if (
            request_type != "invalid"
            and retrieval is not None
            and retrieval_output.confidence < INVALID_RETRIEVAL_CONFIDENCE_MAX
            and corpus_token_overlap_ratio(
                ticket.issue,
                ticket.subject,
                retrieval.chunk.text,
            )
            < INVALID_CORPUS_OVERLAP_MAX
        ):
            trace = self._decision_trace(
                detected_company=inferred_company,
                retrieval_output=retrieval_output,
                pipeline_stage=pipeline_stage,
                risk_flags=["invalid_low_corpus_match"],
                multi_intent=multi_intent,
                final_status="replied",
            )
            return Prediction(
                status="replied",
                product_area=product_area,
                response=INVALID_REPLY,
                justification=self._justification_from_trace(
                    trace,
                    retrieval=retrieval,
                    request_type="invalid",
                    product_area=product_area,
                    escalation_reason="low_corpus_relevance",
                ),
                request_type="invalid",
            )

        # Overlap for risk (full chunk for stable ratio)
        overlap = corpus_token_overlap_ratio(
            ticket.issue,
            ticket.subject,
            retrieval.chunk.text if retrieval is not None else None,
        )

        risk_intents = self.classifier.detect_intents(ticket.issue, ticket.subject)
        escalation_intents = self.classifier.merge_intents_for_escalation(risk_intents, pipeline_intents)

        # 4–5) Risk evaluation + escalation decision
        assessment = evaluate_risk(
            ticket.issue,
            ticket.subject,
            retrieval_output.confidence,
            detected_domains,
            escalation_intents,
            corpus_token_overlap_ratio=overlap,
            multi_intent_pipeline=multi_intent,
        )
        should_escalate = assessment.escalate
        escalation_reason = assessment.justification
        risk_flags = list(assessment.risk_flags)

        # 6–7) Response + justification
        if should_escalate:
            response = self._escalated_response(retrieval, inferred_company)
            status = "escalated"
        else:
            response = self._grounded_response(retrieval, inferred_company)
            status = "replied"

        trace = self._decision_trace(
            detected_company=inferred_company,
            retrieval_output=retrieval_output,
            pipeline_stage=pipeline_stage,
            risk_flags=risk_flags,
            multi_intent=multi_intent,
            final_status=status,
        )
        justification = self._justification_from_trace(
            trace,
            retrieval=retrieval,
            request_type=request_type,
            product_area=product_area,
            escalation_reason=escalation_reason,
        )

        return Prediction(
            status=status,
            product_area=product_area,
            response=response,
            justification=justification,
            request_type=request_type,
        )

    @staticmethod
    def _is_invalid_ticket_early(ticket: TicketInput) -> bool:
        issue = (ticket.issue or "").strip()
        combined = normalize_text(f"{ticket.subject} {ticket.issue}")
        toks = tokenize(combined)
        if len(issue) < MIN_ISSUE_CHAR_THRESHOLD:
            return True
        if len(toks) < MIN_TICKET_TOKEN_THRESHOLD:
            return True
        return False

    @staticmethod
    def _decision_trace(
        *,
        detected_company: str,
        retrieval_output: RetrievalOutput | None,
        pipeline_stage: str,
        risk_flags: list[str],
        multi_intent: bool,
        final_status: str,
    ) -> dict[str, object]:
        return {
            "detected_company": detected_company,
            "retrieval_confidence": retrieval_output.confidence if retrieval_output else 0.0,
            "retrieval_best_score": retrieval_output.best_score if retrieval_output else 0.0,
            "retrieval_pipeline_stage": pipeline_stage,
            "risk_flags": sorted(risk_flags),
            "multi_intent": multi_intent,
            "final_status": final_status,
        }

    def _grounded_response(self, retrieval: RetrievalResult | None, company: str) -> str:
        if retrieval is None:
            return "I could not find a matching article in our local support corpus. Please contact human support."
        body = (retrieval.snippet or "").strip()
        if not body:
            body = self._excerpt_from_chunk(retrieval.chunk.text)
        prefix = f"Based on our local {company.capitalize() if company != 'none' else ''} support documentation: ".strip()
        return join_nonempty([prefix, body], sep="")

    def _escalated_response(self, retrieval: RetrievalResult | None, company: str) -> str:
        if retrieval is None:
            return (
                "This request needs human support review because I do not have sufficient matching coverage "
                "in the local support corpus."
            )
        body = (retrieval.snippet or "").strip()
        if not body:
            body = self._excerpt_from_chunk(retrieval.chunk.text)
        return (
            "I am escalating this request to a human support specialist due to policy/risk sensitivity. "
            f"Closest local guidance: {body}"
        )

    @staticmethod
    def _excerpt_from_chunk(text: str, max_len: int = 360) -> str:
        cleaned = re.sub(r"\s+", " ", normalize_text(text)).strip()
        if len(cleaned) <= max_len:
            return cleaned
        return f"{cleaned[:max_len].rstrip()}..."

    @staticmethod
    def _human_doc_summary(retrieval: RetrievalResult | None) -> str:
        if retrieval is None:
            return "no matching article"
        ch = retrieval.chunk
        crumbs = " > ".join(ch.breadcrumbs) if ch.breadcrumbs else ""
        folder = " / ".join(ch.product_hints[-2:]) if len(ch.product_hints) >= 2 else ""
        title = (ch.article_title or ch.heading or "").strip()
        parts = [p for p in (crumbs, folder, title) if p]
        if parts:
            return "; ".join(parts)
        return Path(ch.source_path).name

    @staticmethod
    def _snippet_blurb(snippet: str, max_len: int = 220) -> str:
        one_line = re.sub(r"\s+", " ", snippet).strip()
        if len(one_line) <= max_len:
            return one_line
        return f"{one_line[:max_len].rstrip()}..."

    def _justification_from_trace(
        self,
        trace: dict[str, object],
        retrieval: RetrievalResult | None,
        request_type: str,
        product_area: str,
        escalation_reason: str,
    ) -> str:
        doc_line = self._human_doc_summary(retrieval)
        snippet_text = ""
        if retrieval and retrieval.snippet.strip():
            snippet_text = self._snippet_blurb(retrieval.snippet)
        elif retrieval:
            snippet_text = self._snippet_blurb(retrieval.chunk.text)
        return (
            f"Trace: company={trace['detected_company']}, "
            f"retrieval_confidence={trace['retrieval_confidence']:.3f}, "
            f"pipeline={trace['retrieval_pipeline_stage']}, "
            f"multi_intent={trace['multi_intent']}, "
            f"risk_flags={trace['risk_flags']}, "
            f"status={trace['final_status']}. "
            f"Classified {request_type} / {product_area}. "
            f"Escalation policy: {escalation_reason}. "
            f"Response generated from documentation context ({doc_line}), "
            f"grounded on excerpt: {snippet_text}"
        )
