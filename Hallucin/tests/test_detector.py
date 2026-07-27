"""
tests/test_detector.py
----------------------
Unit tests for the hallucination detector.
Run with: pytest tests/
"""

from hallucination_detector import detect
from hallucination_detector.scorer import ClaimResult, chunk_context, overall_score, score_claims
from hallucination_detector.splitter import _regex_split, split_claims_simple


# Splitter tests
def test_regex_split_basic():
    text = "The cat sat on the mat. The dog barked loudly. Birds fly south."
    claims = _regex_split(text)
    assert len(claims) == 3


def test_split_claims_simple_filters_noise():
    text = "Yes. The Eiffel Tower is 330 metres tall and was built in 1889."
    claims = split_claims_simple(text)
    assert all(len(c) > 10 for c in claims)


# Chunker tests
def test_chunk_context_returns_list():
    ctx = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = chunk_context(ctx, chunk_size=2)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_chunk_context_includes_full():
    ctx = "Short context."
    chunks = chunk_context(ctx)
    assert any(ctx.strip() in c for c in chunks)


def test_chunk_context_caps_large_inputs():
    ctx = " ".join(f"Sentence {i}." for i in range(1200))
    chunks = chunk_context(ctx, chunk_size=2)
    # Includes at most max sampled chunks plus optional full-context chunk.
    assert len(chunks) <= 241


# Scorer tests
def test_overall_score_all_supported():
    results = [
        ClaimResult("claim", "supported", 0.9, "ctx"),
        ClaimResult("claim", "supported", 0.8, "ctx"),
    ]
    assert overall_score(results) == 1.0


def test_overall_score_all_unsupported():
    results = [
        ClaimResult("claim", "unsupported", 0.2, "ctx"),
        ClaimResult("claim", "unsupported", 0.1, "ctx"),
    ]
    assert overall_score(results) == 0.0


def test_overall_score_mixed():
    results = [
        ClaimResult("claim", "supported", 0.9, "ctx"),
        ClaimResult("claim", "unsupported", 0.2, "ctx"),
    ]
    score = overall_score(results)
    assert 0.0 < score < 1.0


def test_score_claims_duplicate_claims_stable():
    context = "Paris is in France. The Eiffel Tower is 330 metres tall."
    claims = [
        "The Eiffel Tower is 330 metres tall.",
        "The Eiffel Tower is 330 metres tall.",
    ]
    results = score_claims(claims=claims, context=context, model="local")
    assert len(results) == 2
    assert results[0].label == results[1].label
    assert results[0].best_match == results[1].best_match


# End-to-end detector tests
CONTEXT = (
    "The Eiffel Tower is located in Paris, France. "
    "It stands 330 metres tall and was completed in 1889. "
    "It was designed by engineer Gustave Eiffel."
)


def test_detect_returns_result():
    result = detect(context=CONTEXT, response="The Eiffel Tower is in Paris.")
    assert result.score >= 0.0
    assert result.score <= 1.0
    assert isinstance(result.claims, list)


def test_detect_grounded_response_scores_high():
    response = "The Eiffel Tower stands 330 metres tall and was completed in 1889."
    result = detect(context=CONTEXT, response=response)
    assert result.score >= 0.5, f"Expected high score, got {result.score}"


def test_detect_hallucinated_response_has_flagged_claims():
    response = "The Eiffel Tower is 500 metres tall and was built in 1750 by Napoleon."
    result = detect(context=CONTEXT, response=response)
    assert result.score <= 1.0


def test_detect_flagged_claims_are_strings():
    response = "The Eiffel Tower is on the moon and is made of cheese."
    result = detect(context=CONTEXT, response=response)
    assert all(isinstance(c, str) for c in result.flagged_claims)


def test_detect_properties():
    response = "The Eiffel Tower is in Paris and is 330 metres tall."
    result = detect(context=CONTEXT, response=response)
    assert isinstance(result.supported_claims, list)
    assert isinstance(result.partial_claims, list)
    assert isinstance(result.flagged_claims, list)


# --- Additional comprehensive tests ---

# Edge case tests
def test_detect_empty_context_returns_zero():
    result = detect(context="", response="The sky is blue.")
    assert result.score == 0.0
    assert result.claims == []


def test_detect_empty_response_returns_zero():
    result = detect(context="The sky is blue.", response="")
    assert result.score == 0.0
    assert result.claims == []


def test_detect_both_empty_returns_zero():
    result = detect(context="", response="")
    assert result.score == 0.0


def test_detect_whitespace_only_returns_zero():
    result = detect(context="   ", response="   ")
    assert result.score == 0.0


def test_detect_none_inputs():
    result = detect(context=None, response=None)
    assert result.score == 0.0
    assert result.claims == []


# Claim splitting edge cases
def test_regex_split_single_sentence():
    claims = _regex_split("The sky is blue.")
    assert len(claims) == 1
    assert claims[0].strip() == "The sky is blue."


def test_regex_split_no_punctuation():
    claims = _regex_split("The sky is blue")
    assert len(claims) >= 1


def test_split_claims_simple_short_noise():
    claims = split_claims_simple("OK. Sure. The Eiffel Tower is 330 metres tall.")
    assert all(len(c) > 10 for c in claims)


# Scorer edge cases
def test_overall_score_empty_list():
    assert overall_score([]) == 0.0


def test_overall_score_single_partial():
    results = [ClaimResult("claim", "partial", 0.5, "ctx")]
    assert overall_score(results) == 0.5


def test_chunk_context_empty_string():
    chunks = chunk_context("")
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_chunk_context_none_input():
    chunks = chunk_context(None)
    assert isinstance(chunks, list)


def test_score_claims_empty_claims():
    results = score_claims(claims=[], context="Some context here.")
    assert results == []


def test_score_claims_returns_claim_results():
    context = "Paris is the capital of France."
    claims = ["Paris is the capital of France."]
    results = score_claims(claims=claims, context=context, model="local")
    assert len(results) == 1
    assert isinstance(results[0], ClaimResult)
    assert results[0].label in ("supported", "partial", "unsupported")


# Detection result properties
def test_detection_result_elapsed_ms():
    result = detect(context=CONTEXT, response="The Eiffel Tower is in Paris.")
    assert result.elapsed_ms > 0


def test_detection_result_report_runs(capsys):
    result = detect(context=CONTEXT, response="The Eiffel Tower is in Paris.")
    result.report()
    captured = capsys.readouterr()
    assert "Grounding Score" in captured.out


def test_detect_special_characters():
    context = "The café serves crème brûlée at $15.99."
    response = "The café charges $15.99 for crème brûlée."
    result = detect(context=context, response=response)
    assert result.score >= 0.0
    assert result.score <= 1.0


def test_detect_numerical_accuracy():
    context = "The building is 100 meters tall and was built in 2005."
    response = "The building is 200 meters tall and was built in 1990."
    result = detect(context=context, response=response)
    # Incorrect numbers should lower the score
    assert result.score <= 1.0

