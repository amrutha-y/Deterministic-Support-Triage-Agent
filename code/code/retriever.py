"""Deterministic multi-stage retriever with optional embedding support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from utils import CorpusChunk, RetrievalResult, company_slug, getenv_flag, tokenize

MIN_SIMILARITY_THRESHOLD = 2.5
SENSITIVE_TERMS = {
    "fraud",
    "stolen",
    "security",
    "billing",
    "refund",
    "access",
    "password",
    "permission",
    "vulnerability",
    "compromised",
}


@dataclass
class _KeywordProfile:
    query_tokens: set[str]
    query_bigrams: set[str]


@dataclass(frozen=True)
class RetrievalOutput:
    """Structured output from multi-stage retrieval."""

    results: list[RetrievalResult]
    """Normalized confidence in [0, 1]: best_raw_score / max_possible_score_estimate."""
    confidence: float
    """Raw best score before gating (for threshold checks and traceability)."""
    best_score: float
    """True if candidates were restricted to inferred company (not global fallback)."""
    domain_filtered: bool


class DeterministicRetriever:
    """Deterministic multi-stage retrieval: company filter, rank, threshold, top-k."""

    def __init__(self, chunks: list[CorpusChunk]) -> None:
        self.chunks = chunks
        self._embeddings_enabled = False
        self._embedder = None
        self._vectors_by_chunk_id: dict[str, object] = {}
        self._enable_embeddings_if_available()

    def _enable_embeddings_if_available(self) -> None:
        if not getenv_flag("ENABLE_EMBEDDINGS", default=False):
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            return
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._embedder = SentenceTransformer(model_name)
        vectors = self._embedder.encode([chunk.text for chunk in self.chunks], normalize_embeddings=True)
        self._vectors_by_chunk_id = {chunk.chunk_id: vectors[idx] for idx, chunk in enumerate(self.chunks)}
        self._embeddings_enabled = True

    @staticmethod
    def _bigrams(tokens: list[str]) -> set[str]:
        return {f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)}

    def _build_profile(self, query: str) -> _KeywordProfile:
        tokens = tokenize(query)
        return _KeywordProfile(query_tokens=set(tokens), query_bigrams=self._bigrams(tokens))

    @staticmethod
    def _normalize_query(subject: str, issue: str, company: str) -> str:
        return " ".join(part for part in (subject, issue, company) if part and str(part).strip())

    def _stage_candidates(self, inferred_company: str) -> tuple[list[CorpusChunk], bool]:
        """Stage 1: company-matched chunks; if none, stage 2: full corpus (fallback)."""
        if inferred_company == "none":
            return self.chunks, False
        filtered = [c for c in self.chunks if c.company == inferred_company]
        if filtered:
            return filtered, True
        return self.chunks, False

    def _sensitive_context_tokens(self, chunk: CorpusChunk) -> set[str]:
        """Tokens from section title, path, and breadcrumbs only — not body text."""
        hint_text = " ".join(chunk.product_hints)
        bc_text = " ".join(chunk.breadcrumbs)
        return (
            set(tokenize(hint_text))
            | set(tokenize(chunk.heading))
            | set(tokenize(bc_text))
        )

    def _keyword_score(self, query: str, chunk: CorpusChunk, inferred_company: str) -> float:
        profile = self._build_profile(query)
        ordered_chunk_tokens = tokenize(chunk.text)
        chunk_tokens = set(ordered_chunk_tokens)
        chunk_bigrams = self._bigrams(ordered_chunk_tokens)

        token_overlap = len(profile.query_tokens & chunk_tokens)
        bigram_overlap = len(profile.query_bigrams & chunk_bigrams)
        score = float(token_overlap) + (2.0 * float(bigram_overlap))

        if inferred_company != "none" and chunk.company == inferred_company:
            score += 3.0

        sensitive_in_query = profile.query_tokens & SENSITIVE_TERMS
        sensitive_context = self._sensitive_context_tokens(chunk)
        contextual_sensitive_hits = len(sensitive_in_query & sensitive_context)
        score += float(contextual_sensitive_hits) * 1.5

        hint_tokens = set(tokenize(" ".join(chunk.product_hints)))
        bc_tokens = set(tokenize(" ".join(chunk.breadcrumbs)))
        hint_overlap = len(profile.query_tokens & (hint_tokens | bc_tokens))
        score += float(hint_overlap) * 0.5
        return score

    def _embedding_scores(self, query: str, candidates: list[CorpusChunk]) -> dict[str, float]:
        assert self._embedder is not None
        query_vector = self._embedder.encode([query], normalize_embeddings=True)[0]
        out: dict[str, float] = {}
        for chunk in candidates:
            vector = self._vectors_by_chunk_id.get(chunk.chunk_id)
            if vector is None:
                continue
            out[chunk.chunk_id] = float((vector * query_vector).sum())
        return out

    def extract_best_snippet(self, chunk: CorpusChunk, query_tokens: set[str], query_bigrams: set[str]) -> str:
        """Best paragraph by token_overlap*2 + bigram_overlap*3; tie-break by index."""
        paragraphs = [p.strip() for p in chunk.text.split("\n\n") if p.strip()]
        if not paragraphs:
            return chunk.text.strip()

        ranked: list[tuple[float, int, str]] = []
        for idx, paragraph in enumerate(paragraphs):
            paragraph_tokens_list = tokenize(paragraph)
            paragraph_tokens = set(paragraph_tokens_list)
            paragraph_bigrams = self._bigrams(paragraph_tokens_list)
            token_overlap = len(query_tokens & paragraph_tokens)
            bigram_overlap = len(query_bigrams & paragraph_bigrams)
            score = float(token_overlap * 2) + float(bigram_overlap * 3)
            ranked.append((score, idx, paragraph))

        ranked.sort(key=lambda row: (-row[0], row[1]))
        return ranked[0][2]

    def _max_possible_score_estimate(self, profile: _KeywordProfile, inferred_company: str) -> float:
        """Upper bound for normalizing raw scores (deterministic, per-query)."""
        qt = profile.query_tokens
        qb = profile.query_bigrams
        # Lexical stack: token + bigram terms, company boost, hint overlap, sensitive-in-context
        base = float(len(qt)) + 2.0 * float(len(qb))
        base += 3.0 if inferred_company != "none" else 0.0
        base += 0.5 * float(len(qt))
        base += 1.5 * float(len(qt & SENSITIVE_TERMS))
        if self._embeddings_enabled:
            base += 5.0
        return base if base > 0 else 1.0

    def _rank_candidates(
        self,
        candidates: list[CorpusChunk],
        query: str,
        inferred_company: str,
    ) -> list[tuple[float, str, str, RetrievalResult]]:
        """Score and rank candidates with deterministic tie-breakers."""
        if not query or not candidates:
            return []

        profile = self._build_profile(query)
        emb_by_id = self._embedding_scores(query, candidates) if self._embeddings_enabled else {}

        ranked: list[tuple[float, str, str, RetrievalResult]] = []
        for chunk in candidates:
            keyword_score = self._keyword_score(query, chunk, inferred_company)
            if self._embeddings_enabled:
                embedding_score = emb_by_id.get(chunk.chunk_id, 0.0)
                final_score = keyword_score + (embedding_score * 5.0)
                method = "hybrid"
                explanation = "embedding_similarity + keyword_overlap"
            else:
                final_score = keyword_score
                method = "keyword"
                explanation = "keyword_overlap + company_match + hint_overlap"

            snippet = self.extract_best_snippet(chunk, profile.query_tokens, profile.query_bigrams)
            result = RetrievalResult(
                chunk=chunk,
                score=final_score,
                method=method,
                snippet=snippet,
                explanation=explanation,
            )
            ranked.append((final_score, chunk.source_path, chunk.chunk_id, result))

        ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
        return ranked

    def _retrieve_output(
        self,
        issue: str,
        subject: str,
        company: str,
        top_k: int,
        *,
        scope: str = "domain_first",
    ) -> RetrievalOutput:
        """scope: ``domain_first`` (company filter then fallback inside ranking set) or ``global_only``."""
        query = self._normalize_query(subject, issue, company)
        if not query:
            return RetrievalOutput(
                results=[],
                confidence=0.0,
                best_score=0.0,
                domain_filtered=False,
            )

        inferred_company = company_slug(company)
        if scope == "global_only":
            candidates = self.chunks
            domain_filtered = False
        else:
            candidates, domain_filtered = self._stage_candidates(inferred_company)
        profile = self._build_profile(query)
        max_possible = self._max_possible_score_estimate(profile, inferred_company)

        ranked = self._rank_candidates(candidates, query, inferred_company)
        if not ranked:
            return RetrievalOutput(
                results=[],
                confidence=0.0,
                best_score=0.0,
                domain_filtered=domain_filtered,
            )

        best_raw = ranked[0][0]
        normalized_conf = best_raw / max_possible

        if best_raw < MIN_SIMILARITY_THRESHOLD:
            return RetrievalOutput(
                results=[],
                confidence=normalized_conf,
                best_score=best_raw,
                domain_filtered=domain_filtered,
            )

        top_results = [row[3] for row in ranked[: max(1, top_k)]]
        return RetrievalOutput(
            results=top_results,
            confidence=normalized_conf,
            best_score=best_raw,
            domain_filtered=domain_filtered,
        )

    def retrieve_with_fallback_pipeline(
        self,
        issue: str,
        subject: str,
        company: str,
        top_k: int = 3,
    ) -> tuple[RetrievalOutput, str]:
        """Domain-scoped retrieval first; if no passing hit, search full corpus.

        Returns ``(output, pipeline_stage)`` where ``pipeline_stage`` is
        ``domain``, ``global_fallback``, or ``empty``.
        """
        first = self._retrieve_output(issue, subject, company, top_k=top_k, scope="domain_first")
        if first.results:
            return first, "domain"
        second = self._retrieve_output(issue, subject, company, top_k=top_k, scope="global_only")
        if second.results:
            return second, "global_fallback"
        return second, "empty"

    def retrieve(
        self,
        issue: str,
        subject: str,
        company: str,
        top_k: int = 3,
        **kwargs: Any,
    ) -> RetrievalOutput:
        """Run multi-stage retrieval. Returns ``RetrievalOutput`` only.

        Extra keyword arguments (e.g. ``return_output``) are accepted for
        backward compatibility and ignored.
        """
        _ = kwargs
        return self._retrieve_output(issue, subject, company, top_k=top_k)
