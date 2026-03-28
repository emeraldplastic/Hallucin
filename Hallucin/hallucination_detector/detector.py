from __future__ import annotations
from dataclasses import dataclass, field
from .splitter import split_claims_simple, split_claims_llm
from .scorer import score_claims, overall_score, ClaimResult


@dataclass
class DetectionResult:
    score: float
    claims: list = field(default_factory=list)

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
        icons = {"supported": "✅", "partial": "⚠️ ", "unsupported": "❌"}
        print(f"\n{'='*60}")
        print(f"  Grounding Score: {self.score:.2f} / 1.00")
        print(f"  Claims: {len(self.supported_claims)} supported  {len(self.partial_claims)} partial  {len(self.flagged_claims)} unsupported")
        print(f"{'='*60}")
        for r in self.claims:
            print(f"\n{icons[r.label]} [{r.score:.2f}]  {r.claim}")
            if r.label != "supported":
                print(f"     Best match: {r.best_match[:80]}")
        print()


def detect(context, response, use_llm=False, anthropic_client=None, model_name="all-MiniLM-L6-v2", llm_model="claude-haiku-4-5-20251001"):
    from .scorer import load_model
    if isinstance(model_name, str):
        model = load_model(model_name)
    else:
        model = model_name
    claims = split_claims_simple(response)
    if not claims:
        return DetectionResult(score=0.0, claims=[])
    claim_results = score_claims(claims, context, model=model)
    if anthropic_client is not None:
        claim_results = _llm_recheck(claim_results, context, anthropic_client, llm_model)
    score = overall_score(claim_results)
    return DetectionResult(score=score, claims=claim_results)


def _llm_recheck(claim_results, context, client, model):
    from .scorer import ClaimResult
    rechecked = []
    for r in claim_results:
        if r.label in ("supported", "partial"):
            prompt = (
                f"Given this context:\n{context}\n\n"
                f"Is this claim supported, partial, or unsupported?\n"
                f"Claim: {r.claim}\n"
                f"Reply with one word only: supported, partial, or unsupported."
            )
            message = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            verdict = message.content[0].text.strip().lower()
            if verdict in ("supported", "partial", "unsupported"):
                rechecked.append(ClaimResult(
                    claim=r.claim,
                    label=verdict,
                    score=r.score,
                    best_match=r.best_match,
                ))
                continue
        rechecked.append(r)
    return rechecked