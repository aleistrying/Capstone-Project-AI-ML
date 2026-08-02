"""
Keyword / entity extraction for building focused recommendation queries.

ORIGINAL AUTHOR: Lili Marcela Pérez Clavijo.
This logic originated in Lili's intent-based chatbot prototype (notebooks/lili/).
It is ported here into the project's `src/` so the Streamlit MVP can
reuse it. Ported / adapted by the MVP integration (Alejandro) to:
  - build the genre vocabulary from THIS project's dataset (data/processed),
  - keep using the project's saved TF-IDF model (Lili's model is not used),
  - parse the stringified `genres_list` cleanly instead of whitespace-splitting.

WHY THIS EXISTS
---------------
The old chat path fed the *entire* user sentence into TF-IDF, so filler words
("I would like to see a movie that …") diluted the query vector and weak matches
resulted. Lili's approach strips the sentence down to meaningful content words
and detected genres, producing a tight query string. Crucially, words that carry
meaning but aren't in any predefined list (e.g. "psychological") are KEPT as
keywords and passed straight to the cosine-similarity model — that's exactly the
signal we want the recommender to match on.
"""

import ast
import re

import nltk

nltk.download("stopwords", quiet=True)
_STOPWORDS = set(nltk.corpus.stopwords.words("english"))

# Domain filler words to discard — these say "I want a movie" but carry no signal
# for similarity. Taken from the "movies" intent in Lili's intents.json.
IGNORED_WORDS: set[str] = {
    "movie",
    "movies",
    "film",
    "films",
    "watch",
    "recommend",
    "recommendation",
    "suggest",
    "suggestions",
    "want",
    "see",
}

# Conversational/visual filler NOT caught by stopwords. Added during query tuning
# (Alejandro) to raise match quality: these words rarely correspond to useful
# movie-text signal, so dropping them concentrates the query on real content
# words and measurably increases cosine similarity for relevant films.
QUERY_NOISE: set[str] = {
    "would",
    "like",
    "really",
    "kind",
    "something",
    "anything",
    "maybe",
    "please",
    "includes",
    "include",
    "including",
    "black",
    "white",
    "colour",
    "color",
}

# Thematic synonym expansion. Maps a meaning word the user typed to related terms
# that appear in movie overviews/keywords, so the query overlaps more of the
# relevant documents (higher cosine + better relevance). Added during query
# tuning (Alejandro); expanded terms are filtered to the model vocabulary before
# use so no out-of-vocabulary noise is introduced.
THEMES: dict[str, list[str]] = {
    "psychological": ["psycho", "mind", "obsession", "paranoia", "sanity"],
    "terror": ["horror", "fear", "nightmare", "killer", "evil"],
    "scary": ["horror", "fear", "nightmare", "creepy"],
    "horror": ["fear", "nightmare", "evil", "killer"],
    "funny": ["comedy", "hilarious", "laugh", "humor"],
    "comedy": ["funny", "hilarious", "laugh"],
    "romantic": ["romance", "love", "relationship"],
    "romance": ["love", "relationship", "romantic"],
    "dark": ["grim", "gritty", "noir"],
    "intense": ["thrilling", "suspense"],
    "space": ["alien", "galaxy", "planet"],
    "action": ["fight", "chase", "explosion"],
}


def _parse_genres_cell(raw) -> list[str]:
    """Parse one `genres_list` cell (a stringified list) into lowercase genre names."""
    if isinstance(raw, list):
        return [str(g).lower() for g in raw]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return [str(g).lower() for g in parsed]
        except (ValueError, SyntaxError):
            return [g.strip().lower() for g in raw.split(",") if g.strip()]
    return []


def build_genre_vocabulary(movies_df) -> set[str]:
    """
    Build the set of genre names that actually appear in the dataset.

    Lili's original derived genres from `genres_list`; here we parse the column
    properly so multi-word genres like "science fiction" survive intact.
    """
    vocab: set[str] = set()
    for cell in movies_df["genres_list"].dropna():
        vocab.update(_parse_genres_cell(cell))
    return {g for g in vocab if g}


def extract_entities(text: str, genre_vocab: set[str]) -> dict:
    """
    Detect genres (from the dataset vocabulary) and explicit 4-digit years.

    AUTHOR: Lili — logic ported from MovieBot.extract_entities.
    """
    text = text.lower()
    entities: dict[str, list] = {"genres": [], "years": []}

    for genre in genre_vocab:
        # word-boundary match so "war" doesn't fire inside "warm", etc.
        if re.search(rf"\b{re.escape(genre)}\b", text):
            entities["genres"].append(genre)

    entities["years"] = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    return entities


def extract_keywords(text: str, entities: dict) -> list[str]:
    """
    Reduce the sentence to meaningful content words.

    AUTHOR: Lili — logic ported from MovieBot.extract_keywords.
    Drops detected genres/years, English stopwords, domain filler (IGNORED_WORDS)
    and very short tokens. Everything else — including words with no predefined
    meaning in our lists, like "psychological" — is KEPT and later handed to the
    cosine-similarity model, because those words carry the real intent.
    """
    text = text.lower()

    for genre in entities["genres"]:
        text = text.replace(genre, "")
    for year in entities["years"]:
        text = text.replace(year, "")

    words = re.findall(r"\w+", text)
    return [
        word
        for word in words
        if word not in _STOPWORDS
        and word not in IGNORED_WORDS
        and word not in QUERY_NOISE
        and len(word) > 2
        # Drop decade/year tokens like "90s", "1990", "2000s": these are handled
        # by the year filter, not by text similarity.
        and not re.fullmatch(r"\d{2,4}s?", word)
    ]


def expand_terms(terms: list[str], vocab: set[str] | None = None) -> list[str]:
    """
    Add thematic synonyms for recognised meaning words (see THEMES).

    Tuning addition (Alejandro): expanding meaning words to related vocabulary
    increases overlap with relevant movie text, which raises cosine similarity
    and relevance. If `vocab` (the model's known terms) is given, expanded terms
    not in it are dropped so we never add out-of-vocabulary noise.
    """
    out = list(terms)
    for w in terms:
        out.extend(THEMES.get(w, []))
    if vocab is not None:
        out = [w for w in out if w in vocab]
    # De-duplicate, preserve order.
    return list(dict.fromkeys(out))


def build_query(
    text: str, genre_vocab: set[str], vocab: set[str] | None = None
) -> dict:
    """
    Turn raw user text into a focused query string for TF-IDF.

    AUTHOR: Lili — base logic ported from MovieBot.build_query.
    Query tuning (denoising + thematic expansion) added by Alejandro to raise
    match quality; pass `vocab` (vectorizer.vocabulary_) to keep expanded terms
    within the model's known words.

    Returns:
        {
          "entities": {"genres": [...], "years": [...]},
          "keywords": [...],        # meaningful content words
          "query":    "genre1 genre2 keyword1 ... synonym1 ...",
        }
    """
    entities = extract_entities(text, genre_vocab)
    keywords = extract_keywords(text, entities)

    query_parts = entities["genres"] + keywords
    query_parts = expand_terms(query_parts, vocab)

    return {
        "entities": entities,
        "keywords": keywords,
        "query": " ".join(query_parts),
    }
