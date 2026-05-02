"""
Scores how well each claim is grounded in the provided context.

Default behavior prefers `sentence-transformers` when available, then
automatically falls back to a fast local hashing encoder when offline.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

# Thresholds tuned for both transformer embeddings and local hash fallback.
SUPPORTED_THRESHOLD = 0.63
PARTIAL_THRESHOLD = 0.42

SupportLabel = Literal["supported", "partial", "unsupported"]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_NEGATION_RE = re.compile(r"\b(?:no|not|never|none|without|cannot|can't|won't|n't)\b")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}

MAX_CONTEXT_CHUNKS = max(1, int(os.getenv("HALLUCIN_MAX_CONTEXT_CHUNKS", "240")))
TOP_MATCH_CANDIDATES = max(1, int(os.getenv("HALLUCIN_TOP_MATCH_CANDIDATES", "3")))
FULL_CONTEXT_APPEND_LIMIT = max(
    0, int(os.getenv("HALLUCIN_FULL_CONTEXT_APPEND_LIMIT", "24000"))
)


@dataclass
class ClaimResult:
    claim: str
    label: SupportLabel
    score: float
    best_match: str


class LocalHashEncoder:
    """
    Lightweight offline embedder using hashed token and char-gram features.
    It is deterministic, fast, and avoids network/model downloads.
    """

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions

    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ):
        matrix = np.vstack([self._encode_one(text) for text in texts]).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        return matrix if convert_to_numpy else matrix.tolist()

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        text = text or ""
        lowered = text.lower()
        tokens = _TOKEN_RE.findall(lowered)

        for token in tokens:
            idx = hash(("tok", token)) % self.dimensions
            vector[idx] += 1.0
            if token.isdigit():
                num_idx = hash(("num", token)) % self.dimensions
                vector[num_idx] += 2.0

            # Add short character grams to better catch name/number variations.
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    gram = token[i : i + 3]
                    gram_idx = hash(("gram", gram)) % self.dimensions
                    vector[gram_idx] += 0.25

        # Add lightweight length signal to separate very short snippets.
        vector[hash(("len", min(len(tokens), 30))) % self.dimensions] += 0.5
        vector[hash(("chars", min(len(lowered), 500))) % self.dimensions] += 0.25
        return vector


@lru_cache(maxsize=4)
def load_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load and cache an embedding model.

    Falls back to a local hashing encoder when transformer loading fails
    (common in offline or restricted environments).
    """
    if model_name in {"local-hash", "local"}:
        return LocalHashEncoder()

    try:
        from sentence_transformers import SentenceTransformer

        try:
            return SentenceTransformer(model_name, local_files_only=True)
        except TypeError:
            # Older versions may not support `local_files_only`.
            return LocalHashEncoder()
    except Exception:
        return LocalHashEncoder()


def chunk_context(context: str, chunk_size: int = 3) -> list[str]:
    """
    Split context into sentence chunks plus one full-context chunk.
    Uses a sliding window for better localized matching.
    """
    clean = (context or "").strip()
    if not clean:
        return [""]
    return list(_chunk_context_cached(clean, max(1, int(chunk_size))))


@lru_cache(maxsize=128)
def _chunk_context_cached(clean: str, chunk_size: int) -> tuple[str, ...]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    if not sentences:
        return (clean,)

    stride = max(1, chunk_size - 1)
    chunks: list[str] = []
    for i in range(0, len(sentences), stride):
        window = sentences[i : i + chunk_size]
        if window:
            chunks.append(" ".join(window))

    if len(chunks) > MAX_CONTEXT_CHUNKS:
        step = max(1, math.ceil(len(chunks) / MAX_CONTEXT_CHUNKS))
        chunks = chunks[::step]

    if FULL_CONTEXT_APPEND_LIMIT and len(clean) <= FULL_CONTEXT_APPEND_LIMIT:
        if clean not in chunks:
            chunks.append(clean)

    return tuple(chunks)


def score_claims(
    claims: list[str],
    context: str,
    model=None,
) -> list[ClaimResult]:
    """Score each claim against context and return structured labels."""
    if not claims:
        return []

    if model is None:
        model = load_model()
    elif isinstance(model, str):
        model = load_model(model)

    chunks = chunk_context(context)
    unique_texts, remap = _dedupe_texts(claims + chunks)
    unique_embeddings = model.encode(
        unique_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = unique_embeddings[remap]
    claim_embeddings = embeddings[: len(claims)]
    chunk_embeddings = embeddings[len(claims) :]

    # Vectorized similarity matrix: rows=claims, cols=chunks.
    sim_matrix = np.matmul(claim_embeddings, chunk_embeddings.T)

    chunk_numbers = [_extract_numbers(chunk) for chunk in chunks]
    chunk_token_sets = [_content_tokens(chunk) for chunk in chunks]
    chunk_negation = [_has_negation(chunk) for chunk in chunks]
    results: list[ClaimResult] = []

    for idx, claim in enumerate(claims):
        sims = sim_matrix[idx]
        claim_tokens = _content_tokens(claim)
        claim_has_negation = _has_negation(claim)
        best_idx, score = _best_match_with_lexical_blend(
            sims=sims,
            claim_tokens=claim_tokens,
            chunk_token_sets=chunk_token_sets,
            claim_has_negation=claim_has_negation,
            chunk_negation=chunk_negation,
        )
        score = _apply_number_penalty(claim, score, chunk_numbers[best_idx])

        results.append(
            ClaimResult(
                claim=claim,
                label=_label(score),
                score=round(score, 4),
                best_match=chunks[best_idx],
            )
        )

    return results


def overall_score(claim_results: list[ClaimResult]) -> float:
    """Aggregate per-claim labels into a document-level grounding score."""
    if not claim_results:
        return 0.0
    weights = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
    total = sum(weights[result.label] for result in claim_results)
    return round(total / len(claim_results), 4)


def _extract_numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text or ""))


def _dedupe_texts(texts: list[str]) -> tuple[list[str], np.ndarray]:
    unique_texts: list[str] = []
    index_by_text: dict[str, int] = {}
    remap = np.zeros(len(texts), dtype=np.int32)

    for idx, text in enumerate(texts):
        cached_index = index_by_text.get(text)
        if cached_index is None:
            cached_index = len(unique_texts)
            unique_texts.append(text)
            index_by_text[text] = cached_index
        remap[idx] = cached_index

    return unique_texts, remap


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall((text or "").lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _has_negation(text: str) -> bool:
    return bool(_NEGATION_RE.search((text or "").lower()))


def _coverage_ratio(claim_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not claim_tokens:
        return 0.0
    overlap = claim_tokens.intersection(chunk_tokens)
    return len(overlap) / len(claim_tokens)


def _best_match_with_lexical_blend(
    sims: np.ndarray,
    claim_tokens: set[str],
    chunk_token_sets: list[set[str]],
    claim_has_negation: bool,
    chunk_negation: list[bool],
) -> tuple[int, float]:
    candidate_count = min(TOP_MATCH_CANDIDATES, sims.shape[0])
    if candidate_count == sims.shape[0]:
        candidate_indices = np.arange(sims.shape[0], dtype=np.int32)
    elif candidate_count == 1:
        candidate_indices = np.array([int(np.argmax(sims))], dtype=np.int32)
    else:
        candidate_indices = np.argpartition(sims, -candidate_count)[-candidate_count:]

    best_idx = int(candidate_indices[0])
    best_score = -1.0
    for candidate in candidate_indices:
        idx = int(candidate)
        semantic = float(sims[idx])
        lexical = _coverage_ratio(claim_tokens, chunk_token_sets[idx])
        blended = 0.84 * semantic + 0.16 * lexical
        if claim_has_negation != chunk_negation[idx]:
            blended *= 0.92

        if blended > best_score:
            best_idx = idx
            best_score = blended

    return best_idx, max(0.0, min(1.0, best_score))


def _apply_number_penalty(claim: str, base_score: float, chunk_nums: set[str]) -> float:
    claim_nums = _extract_numbers(claim)
    if not claim_nums:
        return base_score

    missing = [num for num in claim_nums if num not in chunk_nums]
    if not missing:
        return base_score

    penalty_ratio = min(0.45, 0.18 * len(missing))
    return max(0.0, base_score * (1.0 - penalty_ratio))


def _label(score: float) -> SupportLabel:
    if score >= SUPPORTED_THRESHOLD:
        return "supported"
    if score >= PARTIAL_THRESHOLD:
        return "partial"
    return "unsupported"
