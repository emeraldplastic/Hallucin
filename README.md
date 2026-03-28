# 🔍 Hallucination Detector

A Python library that detects hallucinations in LLM responses by checking each claim against a source context.

---

## ⚠️ Honest Disclaimer

This library has **two modes** with very different accuracy levels. Please read before using:

| Mode | How it works | Accuracy | Cost |
|------|-------------|----------|------|
| **Offline** (default) | Embedding similarity | Good for obvious hallucinations, fooled by number/name swaps | Free |
| **LLM mode** | Claude reads and judges each claim | Catches subtle hallucinations | Needs Anthropic API key |

**Bottom line:** Offline mode is a solid starting point. For production use, enable LLM mode.

---

## How It Works

1. **Split** — breaks the LLM response into individual claims
2. **Embed** — converts each claim and context into vectors using `sentence-transformers`
3. **Compare** — cosine similarity tells us how grounded each claim is
4. **Recheck** *(LLM mode only)* — Claude reads borderline claims and gives a final verdict

---

## Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install -e .
```

---

## Usage

### Offline mode (free, no API key needed)

```python
from hallucination_detector import detect

result = detect(
    context="The Eiffel Tower is 330m tall and was built in 1889.",
    response="The Eiffel Tower is 300m tall, built in 1889 by Gustave Eiffel."
)

print(result.score)           # 0.5
print(result.flagged_claims)  # may miss subtle number swaps
result.report()               # full breakdown
```

### LLM mode (accurate, needs Anthropic API key)

```python
import anthropic
from hallucination_detector import detect

client = anthropic.Anthropic()  # set ANTHROPIC_API_KEY env variable

result = detect(
    context="The Eiffel Tower is 330m tall and was built in 1889.",
    response="The Eiffel Tower is 500m tall, built in 1750 by Leonardo da Vinci.",
    anthropic_client=client,
)

result.report()
# ❌ The Eiffel Tower is 500m tall         ← caught!
# ❌ built in 1750 by Leonardo da Vinci    ← caught!
```

---

## Output

```
============================================================
  Grounding Score: 0.25 / 1.00
  Claims: 0 supported  1 partial  2 unsupported
============================================================

✅ [0.83]  The Eiffel Tower is in Paris.

❌ [0.31]  It was built in 1750 by Leonardo da Vinci.
     Best match: "built by Gustave Eiffel's company between 1887 and 1889..."
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
hallucination_detector/
├── hallucination_detector/
│   ├── __init__.py
│   ├── detector.py      # main API
│   ├── splitter.py      # breaks response into claims
│   └── scorer.py        # embedding + similarity scoring
├── tests/
│   └── test_detector.py
├── examples/
│   └── basic_usage.py
├── requirements.txt
└── README.md
```

---

## Known Limitations

- **Offline mode** cannot reliably catch number swaps (e.g. "330m → 500m") or name swaps (e.g. "Gustave Eiffel → Leonardo da Vinci") because embedding models measure semantic similarity, not factual accuracy
- **LLM mode** requires an Anthropic API key (paid, starts at $5)
- Very long contexts may need chunking for best results

---

## Roadmap

- [ ] Ollama support (free local LLM mode)
- [ ] CLI tool
- [ ] HTML report output with highlighted claims
- [ ] Citation linking (which source sentence supports each claim)
- [ ] Async batch processing

---

## Contributing

PRs welcome! If you add Ollama support or improve the offline accuracy, please open a pull request.
