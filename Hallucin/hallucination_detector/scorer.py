"""
Scores how well each claim is grounded in the provided context.

Default behavior prefers `sentence-transformers` when available, then
automatically falls back to a fast local hashing encoder when offline.
"""

from __future__ import annotations

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

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    if not sentences:
        return [clean]

    chunk_size = max(1, int(chunk_size))
    stride = max(1, chunk_size - 1)
    chunks: list[str] = []
    for i in range(0, len(sentences), stride):
        window = sentences[i : i + chunk_size]
        if window:
            chunks.append(" ".join(window))

    if clean not in chunks:
        chunks.append(clean)
    return chunks


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

    chunks = chunk_context(context)

    all_texts = claims + chunks
    embeddings = model.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)
    claim_embeddings = embeddings[: len(claims)]
    chunk_embeddings = embeddings[len(claims) :]

    # Vectorized similarity matrix: rows=claims, cols=chunks.
    sim_matrix = np.matmul(claim_embeddings, chunk_embeddings.T)

    chunk_numbers = [_extract_numbers(chunk) for chunk in chunks]
    results: list[ClaimResult] = []

    for idx, claim in enumerate(claims):
        sims = sim_matrix[idx]
        best_idx = int(np.argmax(sims))
        score = float(sims[best_idx])
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
