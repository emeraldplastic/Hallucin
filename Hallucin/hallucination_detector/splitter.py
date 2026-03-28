"""
splitter.py
-----------
Breaks an LLM response into atomic, individually verifiable claims.

Strategy:
  1. Fast path  — spaCy sentence segmentation (no API call, works offline)
  2. Rich path  — Anthropic LLM decomposition (better for complex sentences)

The rich path is opt-in via use_llm=True so the library works fully offline
by default.
"""

from __future__ import annotations
import re
from typing import Optional


def split_claims_simple(text: str) -> list[str]:
    """
    Split text into claims using basic sentence segmentation.
    Works offline, no dependencies beyond stdlib.
    Falls back gracefully if spaCy model isn't installed.
    """
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not downloaded yet — fall back to regex splitter
            return _regex_split(text)
        doc = nlp(text)
        claims = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except ImportError:
        claims = _regex_split(text)

    return [c for c in claims if len(c) > 10]  # filter noise


def split_claims_llm(text: str, client, model: str = "claude-sonnet-4-20250514") -> list[str]:
    """
    Use Claude to decompose a response into atomic factual claims.
    Each claim should be independently verifiable.

    Args:
        text:   The LLM response to decompose.
        client: An anthropic.Anthropic() client instance.
        model:  Model to use for decomposition.

    Returns:
        List of atomic claim strings.
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


# ── helpers ──────────────────────────────────────────────────────────────────

def _regex_split(text: str) -> list[str]:
    """Naive sentence splitter using punctuation."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _parse_numbered_list(text: str) -> list[str]:
    """Extract items from a numbered list like '1. Foo\n2. Bar'."""
    lines = text.strip().splitlines()
    claims = []
    for line in lines:
        # Match lines starting with "1." or "1)" etc.
        match = re.match(r'^\s*\d+[\.\)]\s+(.*)', line)
        if match:
            claims.append(match.group(1).strip())
        elif line.strip() and not re.match(r'^\s*\d+', line):
            # Also catch bullet points or plain lines
            cleaned = re.sub(r'^[-•*]\s*', '', line.strip())
            if cleaned:
                claims.append(cleaned)
    return claims
