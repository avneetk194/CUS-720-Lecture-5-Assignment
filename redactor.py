"""This is where we find private names and cover them with labels like [PER].

Think of redaction as putting stickers over words on a page:
1. Find the words and their positions in the ORIGINAL page.
2. Keep the chosen names that meet the confidence rule, plus every email.
3. Put stickers on from right to left, so earlier positions do not move.

Create one Redactor when the application starts and reuse it for each request.
This module never prints, logs, writes, or stores submitted text on the object.
"""

from __future__ import annotations

import math
import operator
import re
from threading import Lock
from typing import Callable, Optional

from config import (
    CHUNK_OVERLAP,
    ENTITY_TYPES,
    MAX_TEXT_LENGTH,
    MODEL_NAME,
    MODEL_REVISION,
    SCORE_THRESHOLD,
)

# This is where we describe common email addresses, such as a.name+lab@site.org.
# The boundaries stop a match from grabbing only part of a longer address.
# This intentionally covers common addresses, not every legal email format.
EMAIL_PATTERN = re.compile(
    r"(?<![\w.!#$%&'*+/=?^`{|}~@-])"
    r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}(?![\w@-])"
)


class Redactor:
    """Find names with a model, find emails with a pattern, and replace both."""

    def __init__(
        self,
        score_threshold: float = SCORE_THRESHOLD,
        *,
        ner_pipeline: Optional[Callable] = None,
    ) -> None:
        """This is where we load the model ONCE, or accept a fake for tests.

        Member 2 can pass app.config["SCORE_THRESHOLD"] as score_threshold.
        A fake ner_pipeline takes text and returns Hugging Face-shaped results.
        """
        if isinstance(score_threshold, bool) or not isinstance(
            score_threshold, (int, float)
        ):
            raise ValueError("score_threshold must be a number between 0 and 1")
        if not math.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be a number between 0 and 1")

        self.score_threshold = float(score_threshold)
        self._model_lock = Lock()

        if ner_pipeline is None:
            # Import here so fake-model tests need no ML packages or downloads.
            # This is the only project module that imports transformers.
            from transformers import pipeline

            ner_pipeline = pipeline(
                "ner",
                model=MODEL_NAME,
                aggregation_strategy="simple",
                revision=MODEL_REVISION,
                framework="pt",
                device=-1,  # Use the CPU, so a GPU is not required.
                use_fast=True,  # Needed for positions and overlapping chunks.
                stride=CHUNK_OVERLAP,
                model_kwargs={"use_safetensors": True},
            )

        self._find_names = ner_pipeline

    def redact(self, text: str, types: Optional[list[str]] = None) -> dict:
        """Return redacted_text, entities, and model in a JSON-ready dictionary.

        None selects all four name types. An empty list selects only emails.
        Positions are zero-based: start is included and end is not included.
        The returned entity text always comes from text[start:end].
        """
        # This is where we check inputs before doing any expensive model work.
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if not 1 <= len(text) <= MAX_TEXT_LENGTH:
            raise ValueError("text must be 1 to 10,000 characters")
        if types is None:
            types = list(ENTITY_TYPES)
        if not isinstance(types, list) or any(
            not isinstance(entity_type, str) or entity_type not in ENTITY_TYPES
            for entity_type in types
        ):
            raise ValueError("types must be a list containing only PER, LOC, ORG, MISC")

        # Step 1: Find every email. Emails are covered even when types is empty.
        entities = [
            {
                "type": "EMAIL",
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "score": 1.0,
            }
            for match in EMAIL_PATTERN.finditer(text)
        ]

        # Step 2: Ask the model for the selected names above the score threshold.
        if types:
            entities.extend(self._find_model_entities(text, types))

        # Step 3: Make sure no two stickers cover the same letters.
        entities = self._merge_overlapping_entities(text, entities)

        # Step 4: Replace from right to left. Changing the end of a string leaves
        # all earlier positions alone, even when a sticker is shorter than a name.
        redacted_text = text
        for entity in reversed(entities):
            redacted_text = (
                redacted_text[: entity["start"]]
                + f"[{entity['type']}]"
                + redacted_text[entity["end"] :]
            )

        return {
            "redacted_text": redacted_text,
            "entities": entities,
            "model": MODEL_NAME,
        }

    def _find_model_entities(self, text: str, types: list[str]) -> list[dict]:
        """This is where model guesses become exact slices of the original text."""
        # Shared Flask requests take turns using the same model. No transcript is
        # kept on the Redactor; these results exist only during this call.
        with self._model_lock:
            predictions = self._find_names(text)

        entities = []
        for prediction in predictions:
            entity_type = prediction["entity_group"]
            if entity_type not in types:
                continue

            # Hugging Face returns NumPy scores. float() makes them JSON-ready.
            score = float(prediction["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise RuntimeError("model returned an invalid confidence score")
            if score < self.score_threshold:
                continue

            # index() accepts Python and NumPy integers, but rejects decimals.
            # Invalid offsets must fail instead of returning a partly redacted text.
            try:
                start = operator.index(prediction["start"])
                end = operator.index(prediction["end"])
            except TypeError:
                raise RuntimeError("model returned invalid character positions") from None
            if (
                isinstance(prediction["start"], bool)
                or isinstance(prediction["end"], bool)
                or not 0 <= start < end <= len(text)
            ):
                raise RuntimeError("model returned invalid character positions")

            entities.append(
                {
                    "type": entity_type,
                    "text": text[start:end],  # Do not use the model's word pieces.
                    "start": start,
                    "end": end,
                    "score": score,
                }
            )
        return entities

    @staticmethod
    def _merge_overlapping_entities(text: str, entities: list[dict]) -> list[dict]:
        """This is where overlapping matches become one safe replacement.

        Cover the whole shared region so no detected letters are left behind.
        EMAIL wins the label; otherwise use the label with the higher score.
        Matches that merely touch are separate, so adjacent names keep labels.
        """
        merged_entities = []
        for entity in sorted(entities, key=lambda item: (item["start"], item["end"])):
            if not merged_entities or entity["start"] >= merged_entities[-1]["end"]:
                merged_entities.append(entity.copy())
                continue

            previous = merged_entities[-1]
            if entity["type"] == "EMAIL" or (
                previous["type"] != "EMAIL" and entity["score"] > previous["score"]
            ):
                previous["type"] = entity["type"]
                previous["score"] = entity["score"]
            previous["end"] = max(previous["end"], entity["end"])
            previous["text"] = text[previous["start"] : previous["end"]]
        return merged_entities
