"""
scorer.py
---------
Scores how well each claim is grounded in the source context.

Approach:
  - Embed the context and each claim using a local sentence-transformer model
  - Compute cosine similarity between claim and context chunks
  - Label each claim as supported / partial / unsupported based on thresholds
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal

# Thresholds (tunable)
SUPPORTED_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.50

SupportLabel = Literal["supported", "partial", "unsupported"]


@dataclass
class ClaimResult:
    claim: str
    label: SupportLabel
    score: float          # highest cosine similarity against any context chunk
    best_match: str       # the context chunk that matched best


def load_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load and cache a sentence-transformer model.
    all-MiniLM-L6-v2 is fast, small (~80MB), and accurate enough for this task.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def chunk_context(context: str, chunk_size: int = 3) -> list[str]:
    """
    Split context into overlapping sentence-level chunks.
    Chunking improves granularity of matching for long documents.
    """
    sentences = [s.strip() for s in context.split('.') if s.strip()]
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = '. '.join(sentences[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    # Always include the full context as one chunk too (catches holistic matches)
    if context.strip() not in chunks:
        chunks.append(context.strip())
    return chunks


def score_claims(
    claims: list[str],
    context: str,
    model=None,
) -> list[ClaimResult]:
    """
    Score each claim against the source context.

    Args:
        claims:  List of atomic claim strings (from splitter.py)
        context: The original source context the LLM was given
        model:   A loaded SentenceTransformer model (loaded fresh if None)

    Returns:
        List of ClaimResult objects, one per claim
    """
    if model is None:
        model = load_model()

    chunks = chunk_context(context)

    # Embed everything in one batch for speed
    all_texts = claims + chunks
    embeddings = model.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)

    claim_embeddings = embeddings[:len(claims)]
    chunk_embeddings = embeddings[len(claims):]

    results = []
    for i, claim in enumerate(claims):
        # Cosine similarity (embeddings are already L2-normalised, so dot product == cosine)
        sims = np.dot(chunk_embeddings, claim_embeddings[i])
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_chunk = chunks[best_idx]

        label = _label(best_score)
        results.append(ClaimResult(
            claim=claim,
            label=label,
            score=round(best_score, 4),
            best_match=best_chunk,
        ))

    return results


def overall_score(claim_results: list[ClaimResult]) -> float:
    """
    Aggregate per-claim scores into a single document-level grounding score (0–1).
    Unsupported claims are penalised heavily.
    """
    if not claim_results:
        return 0.0
    weights = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
    total = sum(weights[r.label] for r in claim_results)
    return round(total / len(claim_results), 4)


def _label(score: float) -> SupportLabel:
    if score >= SUPPORTED_THRESHOLD:
        return "supported"
    elif score >= PARTIAL_THRESHOLD:
        return "partial"
    return "unsupported"
