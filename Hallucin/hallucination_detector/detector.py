from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from .scorer import ClaimResult, overall_score, score_claims
from .splitter import split_claims_llm, split_claims_simple
from .config import Config
from .security import get_security_manager

@dataclass
class DetectionResult:
    score: float
    claims: list[ClaimResult] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def supported_claims(self):
        return [c for c in self.claims if c.label == "supported"]

    @property
    def partial_claims(self):
        return [c for c in self.claims if c.label == "partial"]

    @property
    def flagged_claims(self):
        return [c.claim for c in self.claims if c.label == "unsupported"]

    def report(self):
        icons = {"supported": "[OK]", "partial": "[~]", "unsupported": "[X]"}
        print(f"\n{'=' * 60}")
        print(f"  Grounding Score: {self.score:.2f} / 1.00")
        print(
            f"  Claims: {len(self.supported_claims)} supported  "
            f"{len(self.partial_claims)} partial  {len(self.flagged_claims)} unsupported"
        )
        print(f"  Runtime: {self.elapsed_ms:.1f} ms")
        print(f"{'=' * 60}")
        for result in self.claims:
            print(f"\n{icons[result.label]} [{result.score:.2f}]  {result.claim}")
            if result.label != "supported":
                print(f"     Best match: {result.best_match[:120]}")
        print()


def detect(
    context,
    response,
    use_llm: bool = False,
    anthropic_client=None,
    model_name="local",
    llm_model="claude-haiku-4-5-20251001",
    max_claims: int = None,
):
    from .scorer import load_model

    sm = get_security_manager()
    max_claims = max_claims if max_claims is not None else Config.MAX_CLAIMS

    start = perf_counter()
    context = sm.sanitize((context or "").strip(), max_length=Config.MAX_TEXT_CHARS)
    response = sm.sanitize((response or "").strip(), max_length=Config.MAX_TEXT_CHARS)

    if not context or not response:
        return DetectionResult(score=0.0, claims=[], elapsed_ms=0.0)

    if anthropic_client is not None and use_llm:
        claims = split_claims_llm(response, anthropic_client, model=llm_model)
    else:
        claims = split_claims_simple(response)

    if not claims:
        elapsed_ms = (perf_counter() - start) * 1000.0
        return DetectionResult(score=0.0, claims=[], elapsed_ms=elapsed_ms)

    if len(claims) > max_claims:
        claims = claims[:max_claims]

    model = load_model(model_name) if isinstance(model_name, str) else model_name
    claim_results = score_claims(claims, context, model=model)

    if anthropic_client is not None:
        claim_results = _llm_recheck(claim_results, context, anthropic_client, llm_model)

    elapsed_ms = (perf_counter() - start) * 1000.0
    score = overall_score(claim_results)
    return DetectionResult(score=score, claims=claim_results, elapsed_ms=elapsed_ms)


def _llm_recheck(claim_results, context, client, model):
    sm = get_security_manager()
    rechecked = []
    for result in claim_results:
        if result.label in ("supported", "partial"):
            safe_claim = sm.sanitize(result.claim)
            safe_context = sm.sanitize(context)
            prompt = (
                f"Given this context:\n{safe_context}\n\n"
                f"Is this claim supported, partial, or unsupported?\n"
                f"Claim: {safe_claim}\n"
                "Reply with one word only: supported, partial, or unsupported."
            )
            message = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            verdict = message.content[0].text.strip().lower()
            if verdict in ("supported", "partial", "unsupported"):
                rechecked.append(
                    ClaimResult(
                        claim=result.claim,
                        label=verdict,
                        score=result.score,
                        best_match=result.best_match,
                    )
                )
                continue
        rechecked.append(result)
    return rechecked
