"""This is where we check Member 1's real logic with pretend model guesses.

All transcripts here are invented. No test downloads the Hugging Face model.
Member 3 can add Flask endpoint tests in a separate test_app.py file.
"""

import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from config import CHUNK_OVERLAP, ENTITY_TYPES, MODEL_NAME, MODEL_REVISION
from redactor import Redactor


def model_guess(text, word, entity_type="PER", score=0.99):
    """This is where we give a pretend model the same positions as the real one."""
    start = text.index(word)
    return {
        "entity_group": entity_type,
        "word": "deliberately different model word pieces",
        "start": start,
        "end": start + len(word),
        "score": score,
    }


def test_names_and_email_are_replaced_using_original_positions():
    text = "Maria Lopez works at Riverside Clinic in Denver. maria@example.com"
    fake_model = Mock(return_value=[
        model_guess(text, "Denver", "LOC"),
        model_guess(text, "Maria Lopez"),
        model_guess(text, "Riverside Clinic", "ORG"),
    ])
    result = Redactor(ner_pipeline=fake_model).redact(text)

    assert result["redacted_text"] == "[PER] works at [ORG] in [LOC]. [EMAIL]"
    assert result["model"] == MODEL_NAME
    assert [entity["type"] for entity in result["entities"]] == [
        "PER", "ORG", "LOC", "EMAIL"
    ]
    for entity in result["entities"]:
        assert entity["text"] == text[entity["start"]:entity["end"]]
    fake_model.assert_called_once_with(text)


@pytest.mark.parametrize("entity_type", ENTITY_TYPES)
def test_each_selected_entity_type_is_supported(entity_type):
    fake_model = Mock(return_value=[model_guess("Example", "Example", entity_type)])
    result = Redactor(ner_pipeline=fake_model).redact("Example", [entity_type])
    assert result["redacted_text"] == f"[{entity_type}]"


def test_unselected_types_are_left_alone_but_emails_are_redacted():
    text = "Maria in Denver: maria@example.com"
    fake_model = Mock(return_value=[
        model_guess(text, "Maria"), model_guess(text, "Denver", "LOC")
    ])
    result = Redactor(ner_pipeline=fake_model).redact(text, ["LOC"])
    assert result["redacted_text"] == "Maria in [LOC]: [EMAIL]"


def test_empty_types_redacts_only_emails_without_running_model():
    fake_model = Mock(side_effect=AssertionError("Model should not run"))
    result = Redactor(ner_pipeline=fake_model).redact("Maria: a@example.org", [])
    assert result["redacted_text"] == "Maria: [EMAIL]"
    fake_model.assert_not_called()


@pytest.mark.parametrize("score, expected", [
    (0.7999, "Maria"), (0.80, "[PER]"), (0.8001, "[PER]")
])
def test_default_threshold_includes_exactly_eighty_percent(score, expected):
    fake_model = Mock(return_value=[model_guess("Maria", "Maria", score=score)])
    assert Redactor(ner_pipeline=fake_model).redact("Maria")["redacted_text"] == expected


def test_threshold_can_come_from_application_configuration():
    fake_model = Mock(return_value=[model_guess("Maria", "Maria", score=0.85)])
    redactor = Redactor(score_threshold=0.90, ner_pipeline=fake_model)
    assert redactor.redact("Maria")["entities"] == []


def test_model_number_is_converted_to_json_float():
    # A stand-in for a NumPy scalar: JSON cannot encode this until float() is used.
    class ModelScore:
        def __float__(self):
            return 0.99

    fake_model = Mock(return_value=[model_guess("Maria", "Maria", score=ModelScore())])
    result = Redactor(ner_pipeline=fake_model).redact("Maria")
    assert type(result["entities"][0]["score"]) is float
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize("address", [
    "maria@example.com", "A.Name+lab@dept.example.org", "a_b@example.co.uk",
    "o'connor@example.org", "a@sample-domain.org"
])
def test_common_emails_preserve_surrounding_punctuation(address):
    redactor = Redactor(ner_pipeline=Mock(return_value=[]))
    result = redactor.redact(f"Contact ({address}).", [])
    assert result["redacted_text"] == "Contact ([EMAIL])."
    assert result["entities"][0]["text"] == address


def test_repeated_emails_have_separate_correct_positions():
    text = "a@example.org and a@example.org"
    result = Redactor(ner_pipeline=Mock(return_value=[])).redact(text, [])
    assert result["redacted_text"] == "[EMAIL] and [EMAIL]"
    assert [entity["start"] for entity in result["entities"]] == [0, 18]


def test_email_overlap_removes_the_full_address_once():
    text = "maria@example.com"
    fake_model = Mock(return_value=[model_guess(text, "maria")])
    result = Redactor(ner_pipeline=fake_model).redact(text)
    assert result["redacted_text"] == "[EMAIL]"
    assert len(result["entities"]) == 1
    assert result["entities"][0]["text"] == text


def test_partly_overlapping_names_cover_the_union():
    text = "New York City"
    fake_model = Mock(return_value=[
        model_guess(text, "New York", "LOC", 0.9),
        model_guess(text, "York City", "ORG", 0.95),
    ])
    result = Redactor(ner_pipeline=fake_model).redact(text)
    assert result["redacted_text"] == "[ORG]"
    assert result["entities"][0]["text"] == text


def test_adjacent_matches_stay_separate():
    text = "AB"
    fake_model = Mock(return_value=[model_guess(text, "A"), model_guess(text, "B", "LOC")])
    assert Redactor(ner_pipeline=fake_model).redact(text)["redacted_text"] == "[PER][LOC]"


def test_unicode_positions_and_repeated_names_are_preserved():
    text = "🙂 José met José."
    first = model_guess(text, "José")
    second = dict(first, start=11, end=15)
    fake_model = Mock(return_value=[first, second])
    assert Redactor(ner_pipeline=fake_model).redact(text)["redacted_text"] == "🙂 [PER] met [PER]."


@pytest.mark.parametrize("text", [None, 123, [], "", "x" * 10_001])
def test_invalid_text_fails_before_running_model(text):
    fake_model = Mock()
    with pytest.raises(ValueError, match="text must"):
        Redactor(ner_pipeline=fake_model).redact(text)
    fake_model.assert_not_called()


@pytest.mark.parametrize("types", ["PER", ["EMAIL"], ["per"], [None], [[]], {}, ("PER",)])
def test_invalid_types_fail_before_running_model(types):
    fake_model = Mock()
    with pytest.raises(ValueError, match="types must"):
        Redactor(ner_pipeline=fake_model).redact("Example", types)
    fake_model.assert_not_called()


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True, "0.8"])
def test_invalid_configuration_is_rejected(threshold):
    with pytest.raises(ValueError, match="score_threshold must"):
        Redactor(score_threshold=threshold, ner_pipeline=Mock())


@pytest.mark.parametrize("start, end", [(-1, 4), (0, 6), (3, 3), (0.5, 4), (None, 4)])
def test_invalid_model_positions_raise_a_plain_error(start, end):
    prediction = dict(model_guess("Maria", "Maria"), start=start, end=end)
    with pytest.raises(RuntimeError, match="model returned invalid character positions"):
        Redactor(ner_pipeline=Mock(return_value=[prediction])).redact("Maria")


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_model_scores_are_rejected(score):
    fake_model = Mock(return_value=[model_guess("Maria", "Maria", score=score)])
    with pytest.raises(RuntimeError, match="invalid confidence score"):
        Redactor(ner_pipeline=fake_model).redact("Maria")


def test_maximum_length_text_is_accepted():
    text = "x" * 10_000
    result = Redactor(ner_pipeline=Mock(return_value=[])).redact(text)
    assert result["redacted_text"] == text


def test_no_matches_returns_text_unchanged():
    result = Redactor(ner_pipeline=Mock(return_value=[])).redact("A quiet day.")
    assert result == {"redacted_text": "A quiet day.", "entities": [], "model": MODEL_NAME}


def test_model_is_created_once_with_pinned_revision_and_long_text_support(monkeypatch):
    fake_model = Mock(return_value=[])
    factory = Mock(return_value=fake_model)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=factory))
    redactor = Redactor()
    redactor.redact("First request.")
    redactor.redact("Second request.")
    factory.assert_called_once_with(
        "ner", model=MODEL_NAME, aggregation_strategy="simple", revision=MODEL_REVISION,
        framework="pt", device=-1, use_fast=True, stride=CHUNK_OVERLAP,
        model_kwargs={"use_safetensors": True},
    )
    assert fake_model.call_count == 2


def test_importing_redactor_does_not_import_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    importlib.reload(sys.modules["redactor"])


def test_redactor_does_not_keep_or_log_transcripts(capsys, caplog):
    # A function rather than a Mock, because Mocks deliberately store arguments.
    redactor = Redactor(ner_pipeline=lambda text: [])
    redactor.redact("Invented private text: a@example.org")
    assert set(vars(redactor)) == {"score_threshold", "_model_lock", "_find_names"}
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert caplog.records == []
