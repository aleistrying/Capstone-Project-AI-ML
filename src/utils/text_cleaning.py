"""
Shared text-cleaning for CineAssist.

This is the SINGLE source of truth for how text is normalized. It is used in
two places that MUST stay identical for cosine similarity to be meaningful:

  1. preprocess.py  — cleans each movie's combined_features before TF-IDF fit.
  2. recommender_engine.py — cleans the user's query before TF-IDF transform.

If the query is cleaned differently from the corpus, query tokens won't match
the vocabulary and similarity scores collapse. Keeping both sides on this one
function guarantees they never drift apart.

NOTE: If you change anything here (including toggling the stemmer below), you
must REBUILD the model (run preprocess.py) so the vocabulary matches.
"""

import re
import string

import nltk

nltk.download("stopwords", quiet=True)
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

# Stemming is OPTIONAL and currently OFF. Because this function is shared by the
# corpus build and the query, uncommenting the line below stems BOTH sides
# together (then rerun preprocess.py to rebuild the model).
_stemmer = nltk.PorterStemmer()


def clean_text(text: str) -> str:
    """Lowercase → strip punctuation → tokenize → drop stopwords [→ stem]."""
    text = str(text).lower()
    text = "".join(c for c in text if c not in string.punctuation)
    tokens = re.split(r"\W+", text)
    tokens = [w for w in tokens if w and w not in STOPWORDS]

    # --- Stemmer toggle: uncomment to stem (then rebuild the model) ---
    # tokens = [_stemmer.stem(w) for w in tokens]

    return " ".join(tokens)
