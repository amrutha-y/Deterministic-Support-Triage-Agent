"""Deterministic request and product-area classification."""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

from utils import CorpusChunk, RetrievalResult, company_slug, normalize_text, tokenize


class TicketClassifier:
    """Rule-based classifier for request_type and product_area."""

    COMPANY_KEYWORDS = {
        "hackerrank": {"hackerrank", "assessment", "candidate", "interview", "test"},
        "claude": {"claude", "anthropic", "bedrock", "api key"},
        "visa": {"visa", "card", "merchant", "chargeback", "travelers", "traveller"},
    }

    REQUEST_PATTERNS = OrderedDict(
        [
            ("invalid", {"actor", "movie", "delete all files", "iron man", "weather", "joke"}),
            ("feature_request", {"feature request", "add support", "please add", "can you add", "request new feature"}),
            ("bug", {"down", "not working", "error", "failing", "stopped", "unable", "issue while", "blocked"}),
        ]
    )

    PRODUCT_PATTERNS = OrderedDict(
        [
            ("security", {"security", "vulnerability", "compromised", "fraud", "identity theft"}),
            ("billing", {"billing", "refund", "payment", "charge", "invoice", "subscription"}),
            ("account_access", {"password", "login", "access", "locked", "permission", "role", "seat"}),
            ("interviews", {"interview", "mock interview", "lobby"}),
            ("assessments", {"assessment", "test", "candidate", "submission", "compatibility"}),
            ("privacy", {"privacy", "delete conversation", "data retention", "data use"}),
            ("travel_support", {"travel", "traveller", "stolen card", "lost card", "cash"}),
            ("general_support", {"help", "support"}),
        ]
    )

    def infer_company(self, issue: str, subject: str, company: str) -> str:
        normalized_company = company_slug(company)
        if normalized_company != "none":
            return normalized_company
        text = normalize_text(f"{subject} {issue}")
        for slug, terms in self.COMPANY_KEYWORDS.items():
            if any(term in text for term in terms):
                return slug
        return "none"

    def classify_request_type(self, issue: str, subject: str) -> str:
        text = normalize_text(f"{subject} {issue}")
        if not text:
            return "invalid"
        for request_type, phrases in self.REQUEST_PATTERNS.items():
            if any(phrase in text for phrase in phrases):
                return request_type
        return "product_issue"

    def detect_domains_from_text(self, issue: str, subject: str) -> list[str]:
        """Return sorted unique companies detected from ticket wording."""
        text = normalize_text(f"{subject} {issue}")
        found: set[str] = set()
        for slug, terms in self.COMPANY_KEYWORDS.items():
            if any(term in text for term in terms):
                found.add(slug)
        return sorted(found)

    def detect_domains(self, issue: str, subject: str, company_field: str) -> list[str]:
        """Domains from explicit CSV company plus lexical cues (deterministic, sorted)."""
        domains = set(self.detect_domains_from_text(issue, subject))
        slug = company_slug(company_field)
        if slug != "none":
            domains.add(slug)
        return sorted(domains)

    def detect_intents(self, issue: str, subject: str) -> list[str]:
        """Map ticket text to canonical risk-intent labels for escalation scoring."""
        text = normalize_text(f"{subject} {issue}")
        tokens = set(tokenize(text))
        # Omit overly-generic buckets (e.g. general/help) to avoid every ticket matching.
        risk_areas = ("security", "billing", "account_access", "privacy", "travel_support")
        label_map = {
            "security": "security",
            "billing": "billing",
            "account_access": "account_access",
            "privacy": "privacy",
            "travel_support": "travel_support",
        }
        matched: set[str] = set()
        for area in risk_areas:
            phrases = self.PRODUCT_PATTERNS[area]
            hit = any(phrase in text for phrase in phrases)
            if not hit:
                phrase_tokens = set(token for phrase in phrases for token in tokenize(phrase))
                hit = bool(tokens & phrase_tokens)
            if hit:
                mapped = label_map.get(area)
                if mapped:
                    matched.add(mapped)
        return sorted(matched)

    def detect_pipeline_intent_classes(self, issue: str, subject: str) -> list[str]:
        """Detect coarse intent classes for multi-intent routing (deterministic order)."""
        text = normalize_text(f"{subject} {issue}")
        tokens = set(tokenize(text))
        found: list[str] = []
        rules: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("billing", ("billing", "refund", "invoice", "subscription", "payment", "charge")),
            ("login", ("login", "sign in", "signin", "password", "forgot password")),
            ("access", ("access", "permission", "role", "seat", "workspace", "locked")),
            ("feature_request", ("feature request", "please add", "could you add", "new feature")),
            ("bug", ("not working", "bug", "error", "failed", "down", "broken", "stopped")),
            ("security", ("security", "vulnerability", "fraud", "compromised", "hack")),
        )
        for label, needles in rules:
            hit = False
            for n in needles:
                if " " in n:
                    if n in text:
                        hit = True
                        break
                elif n in tokens:
                    hit = True
                    break
            if hit:
                found.append(label)
        # Dedupe preserving rule order
        seen: set[str] = set()
        ordered: list[str] = []
        for label in found:
            if label not in seen:
                seen.add(label)
                ordered.append(label)
        return ordered

    @staticmethod
    def _slug_area(label: str) -> str:
        s = normalize_text(label)
        s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
        return s[:80] if s else "general_support"

    def product_area_from_chunk_metadata(self, chunk: CorpusChunk) -> str:
        """Derive product_area from breadcrumbs, title, heading, path, filename."""
        if chunk.breadcrumbs:
            return self._slug_area(chunk.breadcrumbs[-1])
        if chunk.article_title.strip():
            return self._slug_area(chunk.article_title)
        if chunk.heading.strip() and chunk.heading != "general":
            return self._slug_area(chunk.heading)
        if chunk.product_hints:
            tail = "_".join(chunk.product_hints[-3:])
            tail = re.sub(r"[^a-z0-9_]+", "_", tail).strip("_")
            if tail:
                return tail[:80]
        stem = Path(chunk.source_path).stem
        stem = re.sub(r"^\d+-", "", stem)
        return self._slug_area(stem.replace("-", " "))

    def classify_product_area(self, issue: str, subject: str, retrieval: RetrievalResult | None) -> str:
        """Prefer ticket keywords; refine using corpus metadata when retrieval exists."""
        text_area = self._classify_product_area_from_ticket(issue, subject)
        if text_area != "general_support":
            return text_area
        if retrieval is not None:
            return self.product_area_from_chunk_metadata(retrieval.chunk)
        return "general_support"

    def _classify_product_area_from_ticket(self, issue: str, subject: str) -> str:
        text = normalize_text(f"{subject} {issue}")
        tokens = set(tokenize(text))
        for area, phrases in self.PRODUCT_PATTERNS.items():
            if any(phrase in text for phrase in phrases):
                return area
            phrase_tokens = set(token for phrase in phrases for token in tokenize(phrase))
            if tokens & phrase_tokens:
                return area
        return "general_support"

    def merge_intents_for_escalation(
        self,
        risk_intents: list[str],
        pipeline_classes: list[str],
    ) -> list[str]:
        """Combine classifier risk intents with pipeline intents for escalation scoring."""
        merged: set[str] = set(risk_intents)
        for c in pipeline_classes:
            if c == "login":
                merged.add("account_access")
            elif c == "access":
                merged.add("account_access")
            elif c == "billing":
                merged.add("billing")
            elif c == "security":
                merged.add("security")
        return sorted(merged)
