"""This is where we keep settings, so changing a rule does not rewrite the code."""

# Model scores run from 0 to 1: 0.80 means 80% confidence, not a score of 80.
SCORE_THRESHOLD = 0.80

# Keep the model version fixed so every teammate downloads the same files.
MODEL_NAME = "dslim/bert-base-NER"
MODEL_REVISION = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"

# These are the four kinds of names the model knows how to find.
ENTITY_TYPES = ("PER", "LOC", "ORG", "MISC")
MAX_TEXT_LENGTH = 10_000

# Long text is read in overlapping pieces. Re-reading 64 tokens gives the model
# some context around each join instead of silently ignoring the end of a text.
CHUNK_OVERLAP = 64
