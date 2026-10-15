import pytest

from app import create_app
from config import SCORE_THRESHOLD
from redactor import Redactor


class FakeRedactor:
    def redact(self, text, types):
        return {
            "redacted_text": "[PER] lives here.",
            "entities": [],
            "model": "fake-model",
        }


def test_health():
    app = create_app(redactor=FakeRedactor())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "ok",
        "model": "dslim/bert-base-NER",
    }


def test_redact_success():
    class RecordingRedactor(FakeRedactor):
        def redact(self, text, types):
            self.received = (text, types)
            return super().redact(text, types)

    fake = RecordingRedactor()
    app = create_app(redactor=fake)
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/api/redact",
        json={"text": "Maria lives here.", "types": ["PER"]},
    )

    assert response.status_code == 200
    assert fake.received == ("Maria lives here.", ["PER"])
    assert response.get_json() == {
        "redacted_text": "[PER] lives here.",
        "entities": [],
        "model": "fake-model",
    }

def test_invalid_entity_type():
    class RejectCallsRedactor:
        def redact(self, text, types):
            raise AssertionError("Invalid input should not reach the redactor")

    app = create_app(redactor=RejectCallsRedactor())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/api/redact",
        json={"text": "Maria lives here.", "types": ["INVALID"]},
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "Invalid entity type."}

def test_empty_text():
    app = create_app(redactor=FakeRedactor())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post("/api/redact", json={"text": ""})

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Text must be 1 to 10,000 characters."
    }

def test_text_too_long():
    app = create_app(redactor=FakeRedactor())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/api/redact",
        json={"text": "a" * 10001},
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Text must be 1 to 10,000 characters."
    }

def test_unknown_api_path():
    app = create_app(redactor=FakeRedactor())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    assert response.get_json() == {"error": "Not found"}

def test_app_passes_threshold_to_redactor(monkeypatch):
    from unittest.mock import Mock

    factory = Mock(return_value=FakeRedactor())
    monkeypatch.setattr("redactor.Redactor", factory)

    app = create_app()
    app.test_client().post("/api/redact", json={"text": "First request."})
    app.test_client().post("/api/redact", json={"text": "Second request."})

    assert app.config["REDACTION_THRESHOLD"] == SCORE_THRESHOLD
    factory.assert_called_once_with(
        score_threshold=app.config["REDACTION_THRESHOLD"]
    )


@pytest.mark.parametrize("payload", [
    [], {"text": 123}, {}, {"text": "Example", "types": None},
    {"text": "Example", "types": "PER"},
    {"text": "Example", "types": [None]},
])
def test_invalid_json_inputs_never_reach_redactor(payload):
    class RejectCallsRedactor:
        def redact(self, text, types):
            raise AssertionError("Invalid input reached the redactor")

    response = create_app(RejectCallsRedactor()).test_client().post(
        "/api/redact", json=payload
    )
    assert response.status_code == 400
    assert set(response.get_json()) == {"error"}


def test_malformed_json_returns_json_error():
    response = create_app(FakeRedactor()).test_client().post(
        "/api/redact", data='{', content_type="application/json"
    )
    assert response.status_code == 400
    assert response.get_json() == {"error": "Request body must be JSON."}


@pytest.mark.parametrize("score, expected", [
    (0.7999, "Maria"), (0.80, "[PER]"), (0.8001, "[PER]")
])
def test_api_enforces_threshold_with_real_redaction_logic(score, expected):
    redactor = Redactor(ner_pipeline=lambda text: [{
        "entity_group": "PER", "start": 0, "end": 5, "score": score,
    }])
    response = create_app(redactor).test_client().post(
        "/api/redact", json={"text": "Maria", "types": ["PER"]}
    )
    assert response.status_code == 200
    assert response.get_json()["redacted_text"] == expected


def test_api_email_only_skips_model_and_preserves_entity_offsets():
    def reject_model(text):
        raise AssertionError("Email-only redaction should skip model inference")

    text = "Maria: maria@example.com"
    response = create_app(Redactor(ner_pipeline=reject_model)).test_client().post(
        "/api/redact", json={"text": text, "types": []}
    )
    assert response.status_code == 200
    result = response.get_json()
    assert result["redacted_text"] == "Maria: [EMAIL]"
    assert result["entities"] == [{
        "type": "EMAIL", "text": "maria@example.com",
        "start": 7, "end": 24, "score": 1.0,
    }]


def test_api_omitted_types_selects_all_names():
    redactor = Redactor(ner_pipeline=lambda text: [{
        "entity_group": "LOC", "start": 0, "end": 6, "score": 0.99,
    }])
    response = create_app(redactor).test_client().post(
        "/api/redact", json={"text": "Denver"}
    )
    assert response.status_code == 200
    assert response.get_json()["redacted_text"] == "[LOC]"


def test_webpage_escapes_transcript_and_handles_email_only_selection():
    redactor = Redactor(ner_pipeline=lambda text: [])
    response = create_app(redactor).test_client().post(
        "/", data={"text": "<script>alert(1)</script> maria@example.com"}
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt; [EMAIL]" in html

def test_webpage_form_submission():
    app = create_app(redactor=FakeRedactor())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/",
        data={"text": "Maria lives here.", "types": "PER"},
    )

    assert response.status_code == 200
    assert response.mimetype == "text/html"
    assert '<p class="result">[PER] lives here.</p>' in response.get_data(
        as_text=True
    )
