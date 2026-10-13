# CUS 720 — Transcript Redaction Service

This project helps a research team remove identifying names from interview
transcripts before a person reviews them for sharing. **Redaction** means
covering private words with labels: `Maria Lopez` becomes `[PER]`, and
`maria@example.com` becomes `[EMAIL]`.

The project now includes the redaction engine, Flask API, webpage,
automated tests, and a Python client example. The components have been
combined and tested on the raphael-tests-client-docs branch, pending
team review before merging into main.

## Redaction engine features

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
├── app.py                   # Flask application and routes
├── redactor.py              # Model loading and redaction
├── config.py                # Redactor settings and model revision
├── templates/
│   └── index.html           # Webpage form and results
├── tests/
│   ├── test_app.py          # Flask API and form tests
│   └── test_redactor.py     # Redaction logic tests
├── client_example.py        # Example HTTP client
├── requirements.txt         # Model, Flask, and client dependencies
├── requirements-dev.txt     # Test dependency
├── README.md                # Setup and usage
├── MEMBER_1.md              # Vincent’s integration notes
├── .gitignore               # Excludes local environments and caches
└── Case Study               # Original assignment
```

| Team member | Responsibility | Current status |
| --- | --- | --- |
| Vincent (Member 1) | ML model, redaction logic, configuration, and redaction tests | 54 tests passed |
| Avneet (Member 2) | Flask application, API routes, and webpage | Integrated on Raphael’s branch |
| Raphael (Member 3) | API/form tests, client example, dependencies, and README | 8 tests passed; real-model client verified; documentation in progress |

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
**Alternative: Conda**

Instead of the virtual environment commands above, you can use:

```sh
conda create -n cus720-w5 python=3.11
conda activate cus720-w5
```

Each teammate uses their own local environment. Integration testing
used Python 3.11.16.

Install the application and model dependencies:

```sh
python -m pip install -r requirements.txt
```

Creating the first real `Redactor()` downloads about **433 MB of model weights**,
plus small configuration/tokenizer files. An internet connection is needed for
that download. Subsequent calls reuse the loaded model; later startups can reuse
Hugging Face's downloaded cache.

## Run the application and example client

From the project root, with your environment active, start Flask:

```sh
python -m flask --app app:create_app run
```

Open http://127.0.0.1:5000 in your browser to use the webpage.
The first startup may download the model. Keep this terminal running.

To try the Python client, open a second terminal in the project root,
activate the same environment, and run:

```sh
python client_example.py
```

The client sends an invented transcript to `/api/redact` and prints
the JSON response as a Python dictionary. Use invented text for this
example because the response includes original detected entities.

Press Ctrl+C in the server terminal to stop Flask.

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

Redactor settings live in [`config.py`](config.py):

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

The model revision and direct dependencies are pinned: `transformers==4.57.1`, `torch==2.8.0`, `Flask==3.1.3`, `requests==2.34.2`, and `pytest==8.4.2`. The weights use the safetensors format. Python versions, hardware, and transitive dependencies can still change numerical
results; identical confidence scores on every machine are not guaranteed.

## Run the tests without downloading the model

With the virtual environment active:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

For a test-only setup, install pytest and Flask:

```sh
python -m pip install -r requirements-dev.txt Flask==3.1.3
python -m pytest -q
```

The tests do not download the model. Redaction tests supply a fake
model to the real Redactor; Flask tests supply a fake redactor or
mock its constructor.

The suite includes 54 redaction test cases and 8 Flask tests covering
health, successful API requests, invalid entity types, empty text,
oversized text, JSON 404 responses, threshold configuration, and
webpage form submission.

**Verification:**
- Full integrated test suite: **62 passed**—54 redaction test cases
  and 8 Flask API/form tests.

- **54 unit test cases passed**, covering confidence boundaries, every supported
  type, email rules, original positions, repeated matches, Unicode, overlaps,
  invalid inputs, JSON conversion, model reuse, and privacy behavior.
- A separate check with the real downloaded model passed for short text, email
  redaction, JSON serialization, and names beyond the first model chunk.
- The installed environment passed `python -m pip check`.

The separate real-model check used invented text. It is not part of the default
pytest suite. Flask endpoint and form tests are implemented in `tests/test_app.py`.
The Python client was also verified against the running service with
the real model using an invented transcript.


## Flask API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Display the transcript form. |
| POST | `/` | Submit the form and display results. |
| GET | `/api/health` | Return service status and model name. |
| POST | `/api/redact` | Redact text from a JSON request. |

Example JSON request to `/api/redact`:

```json
{
  "text": "Maria works in Boston. Email maria@example.com.",
  "types": ["PER", "LOC"]
}
```

Successful requests return HTTP 200 with `redacted_text`, `entities`,
and `model`, as described in the return format above.

Omitting `types` selects all four named-entity types. An empty list
selects email-only redaction. Emails are always checked.

Invalid input returns HTTP 400 with a JSON `error` field.
Unknown API paths return HTTP 404 with `{"error": "Not found"}`.

The application creates one redactor at startup and passes
`app.config["REDACTION_THRESHOLD"]` to its `score_threshold`
parameter. The current threshold is 0.80. Changing it requires
recreating the application.

Tests can supply a fake redactor through `create_app(redactor=...)`
without loading the model.

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
