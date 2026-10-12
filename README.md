# CUS 720 — Transcript Redaction Service

This project helps a research team remove identifying names from interview
transcripts before a person reviews them for sharing. **Redaction** means
covering private words with labels: `Maria Lopez` becomes `[PER]`, and
`maria@example.com` becomes `[EMAIL]`.

**Current contribution: Member 1 — ML and redaction logic is complete.**
The Python redaction engine, configuration, dependencies, and its unit tests are
implemented. The Flask application, webpage, HTTP endpoint tests, and client
example are the next team contributions.

## What I implemented

| Part | What it does |
| --- | --- |
| `Redactor` class | Loads the Hugging Face model when the object is created and reuses it for later calls. |
| Name detection | Uses `dslim/bert-base-NER` with `aggregation_strategy="simple"` to group name pieces. |
| Type selection | Supports people (`PER`), locations (`LOC`), organizations (`ORG`), and miscellaneous names (`MISC`). |
| Confidence rule | Removes model matches with scores **at least 0.80**, including exactly 0.80. The threshold lives in configuration. |
| Email detection | Uses a regular expression to remove common email addresses regardless of the selected name types. |
| Correct positions | Uses character positions from the original input and replaces matches from right to left. |
| Overlap handling | Combines overlapping matches so replacements do not leave parts of a detected name or email behind. |
| Long input handling | Reads long inputs in overlapping model chunks so the end is also processed. |
| JSON-ready results | Converts model scores with `float()` and returns ordinary Python dictionaries and numbers. |
| Input checks | Rejects invalid text, unsupported entity types, and invalid confidence settings. |
| Privacy | Does not print, log, save, or retain submitted transcripts between calls. |

The code uses descriptive names and comments that explain what each section does
and why. The model logic stays in `redactor.py`, ready for the Flask application
to call.

## Files and responsibilities

```text
CUS-720-Lecture-5-Assignment/
├── README.md                 # Project overview and getting started
├── redactor.py               # Member 1: model loading and redaction
├── config.py                 # Shared settings and pinned model version
├── requirements.txt          # Pinned direct ML dependencies
├── requirements-dev.txt      # Pinned test dependency
├── tests/
│   └── test_redactor.py       # Redaction tests with a pretend model
├── MEMBER_1.md               # Detailed teammate integration notes
├── .gitignore                # Excludes environments, caches, and model files
└── Case Study                # Original assignment
```

| Team member | Responsibility | Current status |
| --- | --- | --- |
| Member 1 | ML model, redaction logic, configuration, and supporting unit tests | Implemented and verified |
| Member 2 | `app.py`, `create_app(redactor=None)`, routes, and `templates/index.html` | Planned |
| Member 3 | Flask endpoint tests, client example, and final service documentation/dependencies | Planned |

## Set up the project

Use **Python 3.10–3.12**. Development verification used Python **3.12.13** on
macOS with Apple silicon. A GPU is not required; the model uses the CPU.

Clone the repository if you do not already have it:

```sh
git clone https://github.com/avneetk194/CUS-720-Lecture-5-Assignment.git
cd CUS-720-Lecture-5-Assignment
```

Create a virtual environment: a separate place for this project's Python tools.

**macOS / Linux**, with Python 3.12 installed:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**, with Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the model dependencies:

```sh
python -m pip install -r requirements.txt
```

Creating the first real `Redactor()` downloads about **433 MB of model weights**,
plus small configuration/tokenizer files. An internet connection is needed for
that download. Subsequent calls reuse the loaded model; later startups can reuse
Hugging Face's downloaded cache.

## Use the redactor

Run this in Python from the repository root, with the virtual environment active:

```python
from redactor import Redactor

# Create this once when the application starts.
redactor = Redactor()

# This example text is invented. Never print real research transcripts.
text = "Maria Lopez works in Denver. Email maria@example.com."
result = redactor.redact(text, ["PER", "LOC"])

print(result["redacted_text"])
```

An illustrative result is:

```text
[PER] works in [LOC]. Email [EMAIL].
```

Actual name detections and scores depend on the model. Emails use the regex rule.

### Return format

`redact()` returns a dictionary with three fields: `redacted_text`, `entities`,
and `model`. The following example uses illustrative confidence scores:

```json
{
  "redacted_text": "[PER] works in [LOC]. Email [EMAIL].",
  "entities": [
    {"type": "PER", "text": "Maria Lopez", "start": 0, "end": 11, "score": 0.99},
    {"type": "LOC", "text": "Denver", "start": 21, "end": 27, "score": 0.99},
    {"type": "EMAIL", "text": "maria@example.com", "start": 35, "end": 52, "score": 1.0}
  ],
  "model": "dslim/bert-base-NER"
}
```

Entities are returned in their original order. Positions count from zero.
`start` includes the first character; `end` points just after the last character.
For every returned entity, `text[start:end]` gives its exact original text.
An email's score of `1.0` indicates a regex match, not a model probability.

### Choose what gets covered

| Call | Behavior |
| --- | --- |
| `redactor.redact(text)` | Select all four name types and emails. |
| `redactor.redact(text, None)` | Select all four name types and emails. |
| `redactor.redact(text, ["PER", "LOC"])` | Select people, locations, and emails. |
| `redactor.redact(text, [])` | Cover only emails; skip model inference. |

Text must be a string of **1–10,000 characters**. When provided, `types` must be
a list containing only `PER`, `LOC`, `ORG`, and `MISC`. `EMAIL` is always checked
and should not be included in that list. Invalid inputs raise `ValueError` with
messages that do not repeat the submitted text.

If two matches overlap, the redactor covers their combined range once. An email
match takes priority for the label; otherwise the higher-confidence name label
wins. Matches that only touch remain separate.

## Configuration and repeatable setup

Shared settings live in [`config.py`](config.py):

| Setting | Value | Meaning |
| --- | --- | --- |
| `SCORE_THRESHOLD` | `0.80` | Minimum confidence for model matches; scores run from 0 to 1. |
| `MODEL_NAME` | `dslim/bert-base-NER` | The assignment's named-entity recognition model. |
| `MODEL_REVISION` | `d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc` | Fixed model files for every teammate. |
| `ENTITY_TYPES` | `PER`, `LOC`, `ORG`, `MISC` | Allowed name labels. |
| `MAX_TEXT_LENGTH` | `10_000` | Largest accepted input, in Python string characters. |
| `CHUNK_OVERLAP` | `64` | Tokens re-read between model chunks to preserve context. |

You can pass a configured threshold when creating the object:

```python
redactor = Redactor(score_threshold=0.90)
```

The model revision and direct dependencies are pinned: `transformers==4.57.1`,
`torch==2.8.0`, and `pytest==8.4.2`. The weights use the safetensors format.
Python versions, hardware, and transitive dependencies can still change numerical
results; identical confidence scores on every machine are not guaranteed.

## Run the tests without downloading the model

With the virtual environment active:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

For a test-only setup, installing `requirements-dev.txt` is enough; the tests do
not need the ML dependencies. They pass a pretend model into the **real Redactor**
so the actual filtering and replacement code is exercised.

**Verification completed for this contribution:**

- **54 unit test cases passed**, covering confidence boundaries, every supported
  type, email rules, original positions, repeated matches, Unicode, overlaps,
  invalid inputs, JSON conversion, model reuse, and privacy behavior.
- A separate check with the real downloaded model passed for short text, email
  redaction, JSON serialization, and names beyond the first model chunk.
- The installed environment passed `python -m pip check`.

The separate real-model check used invented text. It is not part of the default
pytest suite. Flask endpoint tests belong in Member 3's `tests/test_app.py`.

## Connect the Flask application

Member 2 should create one Redactor inside `create_app(redactor=None)`, after
loading the application's configuration. Pass
`app.config["SCORE_THRESHOLD"]` into its constructor. Each route then calls:

```python
result = redactor.redact(text, types)
```

Keep the model object for later requests. The constructor accepts
`ner_pipeline=...` for testing this logic; the Flask factory can accept an entire
fake redactor for testing routes. Invalid model scores or positions raise
`RuntimeError`, which the web service should treat as a server failure.

The team's planned web interface is:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`, `POST` | `/` | Transcript form and results, as described in the assignment. |
| `GET` | `/api/health` | Service status and model name. |
| `POST` | `/api/redact` | JSON request containing `text` and optional `types`. |

These routes are the next implementation step. Flask launch commands, HTTP
examples, and the client example should be added when the application is ready.
See [`MEMBER_1.md`](MEMBER_1.md) for detailed integration notes.

## Limitations and privacy

This model provides a first pass for **English text**. It can miss unusual or
lowercase names, ages, dates, phone numbers, job titles, and other identifying
details. The email regex covers common addresses, with limited support for
quoted or internationalized addresses. The confidence cutoff also leaves lower
scoring model matches in the text. **A person must review every transcript before
it is shared.**

Inference runs locally after the model download. The redactor does not send
transcripts to Hugging Face, write them to files, log them, or keep them on the
object between calls.

**The returned `entities` list contains the original removed text** for the
review table. Treat the whole result as sensitive. Request/response logging,
Flask debugging, proxies, and client scripts must preserve the same privacy rule.
The printed example above is for invented text only.

## References

- [Assignment](<Case Study>)
- [Member 1 integration notes](MEMBER_1.md)
- [Hugging Face model card](https://huggingface.co/dslim/bert-base-NER)
- [Hugging Face pipeline documentation](https://huggingface.co/docs/transformers/main_classes/pipelines)
