from app import create_app


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

    factory.assert_called_once_with(
        score_threshold=app.config["REDACTION_THRESHOLD"]
    )

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