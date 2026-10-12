# Member 1: model and redaction logic

`redactor.py` is the part that finds names and covers them. It has no Flask code.
Its module comment explains the whole idea; each section explains one small step.
The files live at the repository root, which is already the project folder.

## Install and try it

Use Python 3.10–3.12. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

In Python:

```python
from redactor import Redactor

# Build this once at startup; reuse it for every transcript.
redactor = Redactor()
result = redactor.redact(
    "Maria Lopez works in Denver. Email maria@example.com.",
    ["PER", "LOC"],
)
```

The first `Redactor()` downloads about 433 MB of model weights into Hugging Face's
cache. After download, the model runs locally on the CPU. Submitted transcripts
are not sent to Hugging Face. Downloaded model files are not transcript files.

## The interface teammates can rely on

```python
{
    "redacted_text": "[PER] works in [LOC]. Email [EMAIL].",
    "entities": [
        {"type": "PER", "text": "Maria Lopez", "start": 0, "end": 11, "score": 0.99},
        {"type": "LOC", "text": "Denver", "start": 21, "end": 27, "score": 0.99},
        {"type": "EMAIL", "text": "maria@example.com", "start": 35, "end": 52, "score": 1.0},
    ],
    "model": "dslim/bert-base-NER",
}
```

The example scores are illustrative. Entities are sorted by original position.
`start` counts from zero; `end` is the first character after the match. In other
words, `text[start:end]` gives exactly the returned entity text. Model scores are
ordinary Python floats, so the result can be returned as JSON.

- `redact(text)` or `redact(text, None)` selects PER, LOC, ORG, and MISC.
- `redact(text, ["PER"])` selects people and still removes every detected email.
- `redact(text, [])` removes only emails and skips model inference.
- Text must be a string with 1–10,000 characters; `types` must be a list containing
  only the four allowed labels. Invalid inputs raise `ValueError` with fixed
  messages that do not include transcript contents.
- Names at or above the configured threshold are removed. `0.80` means 80%.
- Overlapping matches become one replacement covering their whole combined
  range. EMAIL wins the label; otherwise the higher-confidence label wins.
  The resulting entity text and offsets describe that combined range.

## Member 2: connect Flask's configuration

Use the shared setting in the application factory, after loading any app config:

```python
from config import SCORE_THRESHOLD
from redactor import Redactor

app.config.setdefault("SCORE_THRESHOLD", SCORE_THRESHOLD)
if redactor is None:
    redactor = Redactor(score_threshold=app.config["SCORE_THRESHOLD"])
```

Keep that object in the factory's route closure or `app.extensions`. A route
validates the request and calls `redactor.redact(text, types)`; it does not load
the model. Catch input-validation `ValueError`s for JSON 400 responses. Unexpected
model failures are server errors, not bad client requests; invalid model scores
and positions raise `RuntimeError`. Changing app config after startup does not change an
existing Redactor; construct it with the intended threshold at startup.

## Check the logic without downloading the model

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests inject pretend model predictions into the real Redactor. They test
replacement positions, threshold boundaries, selected types, emails, overlaps,
Unicode, validation, JSON numbers, model reuse, and privacy behavior. Member 3
can add endpoint tests in `tests/test_app.py` using the agreed return structure.

The model name, model commit, and direct ML dependencies are pinned. Python,
hardware, and transitive dependencies can still affect numerical results; model
confidence values are not promised to be identical on every machine.

## Limits and privacy

Long inputs use overlapping model chunks rather than silently dropping the end.
This still does not guarantee detection: this English model can miss unusual
names, phone numbers, ages, dates, and other identifying details. The email
pattern covers common addresses; quoted email names and internationalized
addresses are not fully supported. A person must review the result before sharing.

The Redactor does not log or save transcripts or retain them between requests.
The response **does contain the original removed text** in `entities`, as the
assignment requires for the review table. That response is sensitive too: do not
log it, save it automatically, or send it to an unauthorized recipient. Flask,
proxies, and debugging tools must also avoid logging request or response bodies.

Reference: [Hugging Face's model card](https://huggingface.co/dslim/bert-base-NER)
and [token-classification pipeline documentation](https://huggingface.co/docs/transformers/main_classes/pipelines).
