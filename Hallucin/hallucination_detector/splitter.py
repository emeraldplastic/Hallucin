"""
Breaks an LLM response into atomic, individually verifiable claims.

Strategy:
1. Offline-first sentence splitting via cached spaCy pipeline when available
2. Regex fallback when spaCy/model is unavailable
3. Optional Anthropic decomposition for richer claim extraction
"""

from __future__ import annotations

import re
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_spacy_pipeline():
    """Return a cached sentence splitter pipeline, or None when unavailable."""
    try:
        import spacy

        try:
            # Disable heavy components for faster startup and throughput.
            return spacy.load(
                "en_core_web_sm",
                disable=["tagger", "ner", "lemmatizer", "attribute_ruler"],
            )
        except OSError:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp
    except Exception:
        return None


def split_claims_simple(text: str) -> list[str]:
    """Split text into claims quickly with sane noise filtering."""
    text = (text or "").strip()
    if not text:
        return []

    nlp = _get_spacy_pipeline()
    if nlp is not None:
        doc = nlp(text)
        claims = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    else:
        claims = _regex_split(text)

    filtered: list[str] = []
    for claim in claims:
        # Remove tiny acknowledgements/noise while preserving factual short sentences.
        if len(claim) < 6:
            continue
        if sum(1 for c in claim if c.isalnum()) < 4:
            continue
        filtered.append(claim)
    return filtered


def split_claims_llm(text: str, client, model: str = "claude-sonnet-4-20250514") -> list[str]:
    """
    Use Claude to decompose a response into atomic factual claims.
    Each claim should be independently verifiable.
    """
    prompt = f"""Break the following text into a list of simple, atomic, factual claims.
Each claim should be one sentence and independently verifiable.
Return ONLY a numbered list. No preamble or explanation.

Text:
{text}"""

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text
    claims = _parse_numbered_list(raw)
    return [c for c in claims if len(c) > 10]


def _regex_split(text: str) -> list[str]:
    """Naive sentence splitter using punctuation."""
    sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [s.strip() for s in sentences if s.strip()]


def _parse_numbered_list(text: str) -> list[str]:
    """Extract items from a numbered list like '1. Foo\n2. Bar'."""
    lines = (text or "").strip().splitlines()
    claims = []
    for line in lines:
        match = re.match(r"^\s*\d+[\.)]\s+(.*)", line)
        if match:
            claims.append(match.group(1).strip())
        elif line.strip() and not re.match(r"^\s*\d+", line):
            cleaned = re.sub(r"^[-•*]\s*", "", line.strip())
            if cleaned:
                claims.append(cleaned)
    return claims
