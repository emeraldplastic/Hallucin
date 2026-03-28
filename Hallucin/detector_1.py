"""
detector.py
-----------
Public API for the hallucination detector.

Usage:
    from hallucination_detector import detect

    result = detect(
        context="The Eiffel Tower is 330m tall and was built in 1889.",
        response="The Eiffel Tower is 300m tall, built in 1889 by Gustave Eiffel."
    )

    print(result.score)           # 0.61
    print(result.flagged_claims)  # ["The Eiffel Tower is 300m tall"]
    result.report()               # pretty-print full breakdown
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .splitter import split_claims_simple, split_claims_llm
from .scorer import score_claims, overall_score, ClaimResult


@dataclass
class DetectionResult:
    """
    Returned by detect(). Contains the full breakdown of claim-level grounding.
    """
    score: float                          # 0.0 (hallucinated) → 1.0 (fully grounded)
    claims: list[ClaimResult] = field(default_factory=list)

    @property
    def supported_claims(self) -> list[ClaimResult]:
        return [c for c in self.claims if c.label == "supported"]

    @property
    def partial_claims(self) -> list[ClaimResult]:
        return [c for c in self.claims if c.label == "partial"]

    @property
    def flagged_claims(self) -> list[str]:
        """Unsupported claim texts — the likely hallucinations."""
        return [c.claim for c in self.claims if c.label == "unsupported"]

    def report(self) -> None:
        """Pretty-print a human-readable grounding report."""
        label_icons = {"supported": "✅", "partial": "⚠️ ", "unsupported": "❌"}
        print(f"\n{'='*60}")
        print(f"  Grounding Score: {self.score:.2f} / 1.00")
        print(f"  Claims: {len(self.supported_claims)} supported  "
              f"{len(self.partial_claims)} partial  "
              f"{len(self.flagged_claims)} unsupported")
        print(f"{'='*60}")
        for r in self.claims:
            icon = label_icons[r.label]
            print(f"\n{icon} [{r.score:.2f}]  {r.claim}")
            if r.label != "supported":
                print(f"     Best match: \"{r.best_match[:80]}...\"")
        print()


def detect(
    context: str,
    response: str,
    use_llm: bool = False,
    anthropic_client=None,
    model_name: str = "all-MiniLM-L6-v2",
    llm_model: str = "claude-sonnet-4-20250514",
) -> DetectionResult:
    """
    Detect hallucinations in an LLM response against a source context.

    Args:
        context:          The source text the LLM was given (docs, RAG chunks, etc.)
        response:         The LLM's response to verify.
        use_llm:          Use Claude to split claims (richer decomposition).
                          Requires anthropic_client to be set.
        anthropic_client: An anthropic.Anthropic() instance (only needed if use_llm=True).
        model_name:       Sentence-transformer model for embeddings.
        llm_model:        Claude model for claim splitting (if use_llm=True).

    Returns:
        DetectionResult with score, per-claim labels, and flagged claims.
    """
    # Step 1: Split response into claims
    if use_llm and anthropic_client is not None:
        claims = split_claims_llm(response, anthropic_client, model=llm_model)
    else:
        claims = split_claims_simple(response)

    if not claims:
        return DetectionResult(score=0.0, claims=[])

    # Step 2: Score each claim against context
    from .scorer import load_model
    model = load_model(model_name)
    claim_results = score_claims(claims, context, model=model)

    # Step 3: Aggregate
    score = overall_score(claim_results)

    return DetectionResult(score=score, claims=claim_results)
