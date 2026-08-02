"""
Curated parallel evaluation data for CineAssist translation quality.

WHAT THIS DOES
--------------
Holds a small, hand-written parallel corpus of movie-request-style sentences
with human reference translations. It is the "gold standard" that BLEU scoring
in ``translation_quality.py`` compares machine-translation output against.

Every pair is realistic for the CineAssist domain: users asking a chatbot for
film recommendations ("I want a romantic comedy from the 90s", "quiero una
comedia romántica de los 90"). Keeping the eval set in-domain means the BLEU
numbers reflect quality on the text the app actually translates, not generic
news text.

STRUCTURE
---------
``EVAL_PAIRS`` maps a direction key ("es-en") to a list of ``{"src", "ref"}``
dicts, where ``src`` is the source sentence and ``ref`` is the single human
reference translation. sacreBLEU supports multiple references per sentence; we
provide one high-quality reference per pair, which is the common setup for a
curated in-house eval set.

Directions covered:
  es-en  (Spanish  → English)
  en-es  (English  → Spanish)
  fr-en  (French   → English)
  pt-en  (Portuguese → English)

The EN->ES set mirrors the ES->EN set so a reviewer can eyeball both directions,
but the sentences are independent human translations (not a mechanical reverse).
"""

# ---------------------------------------------------------------------------
# Spanish -> English
# ---------------------------------------------------------------------------
ES_EN = [
    {
        "src": "Quiero una comedia romántica de los 90.",
        "ref": "I want a romantic comedy from the 90s.",
    },
    {
        "src": "Recomiéndame una película de acción con muchas explosiones.",
        "ref": "Recommend me an action movie with a lot of explosions.",
    },
    {
        "src": "Busco una película de terror que dé mucho miedo.",
        "ref": "I'm looking for a horror movie that is really scary.",
    },
    {
        "src": "¿Puedes sugerir una película de aventuras para toda la familia?",
        "ref": "Can you suggest an adventure movie for the whole family?",
    },
    {
        "src": "Quiero ver algo divertido esta noche.",
        "ref": "I want to watch something fun tonight.",
    },
    {
        "src": "Muéstrame películas de ciencia ficción sobre viajes espaciales.",
        "ref": "Show me science fiction movies about space travel.",
    },
    {
        "src": "Prefiero películas con una buena historia y personajes profundos.",
        "ref": "I prefer movies with a good story and deep characters.",
    },
    {
        "src": "¿Cuáles son las mejores películas de suspenso de la última década?",
        "ref": "What are the best thriller movies of the last decade?",
    },
    {
        "src": "Me gustan las películas animadas con un final feliz.",
        "ref": "I like animated movies with a happy ending.",
    },
    {
        "src": "Dame una recomendación de película para una tarde tranquila.",
        "ref": "Give me a movie recommendation for a relaxing evening.",
    },
    {
        "src": "Quiero ver una película clásica de los años 80.",
        "ref": "I want to watch a classic movie from the 80s.",
    },
    {
        "src": "Tengo ganas de ver una película oscura e intensa.",
        "ref": "I'm in the mood for a dark and intense movie.",
    },
    {
        "src": "¿Hay buenas películas de fantasía con magia y dragones?",
        "ref": "Are there any good fantasy movies with magic and dragons?",
    },
    {
        "src": "Quiero una película con una calificación superior a ocho.",
        "ref": "I want a movie with a rating above eight.",
    },
    {
        "src": "Me gustaría una película de misterio con giros inesperados.",
        "ref": "I would like a mystery movie with unexpected twists.",
    },
    {
        "src": "Busco un drama basado en hechos reales.",
        "ref": "I'm looking for a drama based on true events.",
    },
]

# ---------------------------------------------------------------------------
# English -> Spanish
# ---------------------------------------------------------------------------
EN_ES = [
    {
        "src": "I want a romantic comedy from the 90s.",
        "ref": "Quiero una comedia romántica de los 90.",
    },
    {
        "src": "Recommend me an action movie with lots of car chases.",
        "ref": "Recomiéndame una película de acción con muchas persecuciones de autos.",
    },
    {
        "src": "I'm looking for a horror film that is genuinely terrifying.",
        "ref": "Busco una película de terror que sea realmente aterradora.",
    },
    {
        "src": "Can you suggest a family-friendly adventure movie?",
        "ref": "¿Puedes sugerir una película de aventuras apta para toda la familia?",
    },
    {
        "src": "I want something funny to watch tonight.",
        "ref": "Quiero ver algo divertido esta noche.",
    },
    {
        "src": "Show me science fiction movies about space exploration.",
        "ref": "Muéstrame películas de ciencia ficción sobre la exploración espacial.",
    },
    {
        "src": "I prefer movies with a good storyline and deep characters.",
        "ref": "Prefiero películas con una buena historia y personajes profundos.",
    },
    {
        "src": "What are the best thriller movies of the last decade?",
        "ref": "¿Cuáles son las mejores películas de suspenso de la última década?",
    },
    {
        "src": "I enjoy animated movies with a feel-good ending.",
        "ref": "Disfruto las películas animadas con un final feliz.",
    },
    {
        "src": "Give me a movie recommendation for a relaxing evening.",
        "ref": "Dame una recomendación de película para una tarde tranquila.",
    },
    {
        "src": "I want to watch a classic movie from the 1980s.",
        "ref": "Quiero ver una película clásica de los años 80.",
    },
    {
        "src": "I'm in the mood for a dark and intense film.",
        "ref": "Tengo ganas de ver una película oscura e intensa.",
    },
    {
        "src": "Are there any good fantasy movies with magic and dragons?",
        "ref": "¿Hay buenas películas de fantasía con magia y dragones?",
    },
    {
        "src": "I want a movie with a rating above eight.",
        "ref": "Quiero una película con una calificación superior a ocho.",
    },
    {
        "src": "I would like a mystery film with unexpected twists.",
        "ref": "Me gustaría una película de misterio con giros inesperados.",
    },
    {
        "src": "I'm looking for a drama based on a true story.",
        "ref": "Busco un drama basado en una historia real.",
    },
]

# ---------------------------------------------------------------------------
# French -> English
# ---------------------------------------------------------------------------
FR_EN = [
    {
        "src": "Je veux une comédie romantique des années 90.",
        "ref": "I want a romantic comedy from the 90s.",
    },
    {
        "src": "Recommande-moi un film d'action avec beaucoup d'explosions.",
        "ref": "Recommend me an action movie with a lot of explosions.",
    },
    {
        "src": "Je cherche un film d'horreur qui fait vraiment peur.",
        "ref": "I'm looking for a horror movie that is really scary.",
    },
    {
        "src": "Peux-tu suggérer un film d'aventure pour toute la famille?",
        "ref": "Can you suggest an adventure movie for the whole family?",
    },
    {
        "src": "Je veux regarder quelque chose de drôle ce soir.",
        "ref": "I want to watch something funny tonight.",
    },
    {
        "src": "Montre-moi des films de science-fiction sur l'exploration spatiale.",
        "ref": "Show me science fiction movies about space exploration.",
    },
    {
        "src": "Je préfère les films avec une bonne histoire et des personnages profonds.",
        "ref": "I prefer movies with a good story and deep characters.",
    },
    {
        "src": "Quels sont les meilleurs thrillers de la dernière décennie?",
        "ref": "What are the best thriller movies of the last decade?",
    },
    {
        "src": "J'aime les films d'animation avec une fin heureuse.",
        "ref": "I like animated movies with a happy ending.",
    },
    {
        "src": "Donne-moi une recommandation de film pour une soirée tranquille.",
        "ref": "Give me a movie recommendation for a quiet evening.",
    },
    {
        "src": "Je veux voir un film classique des années 80.",
        "ref": "I want to watch a classic movie from the 80s.",
    },
    {
        "src": "Y a-t-il de bons films de fantasy avec de la magie et des dragons?",
        "ref": "Are there any good fantasy movies with magic and dragons?",
    },
]

# ---------------------------------------------------------------------------
# Portuguese -> English
# ---------------------------------------------------------------------------
PT_EN = [
    {
        "src": "Quero uma comédia romântica dos anos 90.",
        "ref": "I want a romantic comedy from the 90s.",
    },
    {
        "src": "Recomende-me um filme de ação com muitas explosões.",
        "ref": "Recommend me an action movie with a lot of explosions.",
    },
    {
        "src": "Estou procurando um filme de terror que dê muito medo.",
        "ref": "I'm looking for a horror movie that is really scary.",
    },
    {
        "src": "Você pode sugerir um filme de aventura para toda a família?",
        "ref": "Can you suggest an adventure movie for the whole family?",
    },
    {
        "src": "Quero assistir algo divertido esta noite.",
        "ref": "I want to watch something fun tonight.",
    },
    {
        "src": "Mostre-me filmes de ficção científica sobre exploração espacial.",
        "ref": "Show me science fiction movies about space exploration.",
    },
    {
        "src": "Prefiro filmes com uma boa história e personagens profundos.",
        "ref": "I prefer movies with a good story and deep characters.",
    },
    {
        "src": "Quais são os melhores filmes de suspense da última década?",
        "ref": "What are the best thriller movies of the last decade?",
    },
    {
        "src": "Gosto de filmes de animação com um final feliz.",
        "ref": "I like animated movies with a happy ending.",
    },
    {
        "src": "Me dê uma recomendação de filme para uma noite tranquila.",
        "ref": "Give me a movie recommendation for a quiet evening.",
    },
    {
        "src": "Quero ver um filme clássico dos anos 80.",
        "ref": "I want to watch a classic movie from the 80s.",
    },
    {
        "src": "Há bons filmes de fantasia com magia e dragões?",
        "ref": "Are there any good fantasy movies with magic and dragons?",
    },
]

# ---------------------------------------------------------------------------
# Master registry: direction key -> list of {"src", "ref"} pairs.
# A direction key is "<src_lang>-<tgt_lang>" (e.g. "es-en").
# ---------------------------------------------------------------------------
EVAL_PAIRS = {
    "es-en": ES_EN,
    "en-es": EN_ES,
    "fr-en": FR_EN,
    "pt-en": PT_EN,
}


def get_pairs(direction: str) -> list:
    """Return the list of {"src", "ref"} pairs for a direction key like 'es-en'."""
    if direction not in EVAL_PAIRS:
        raise KeyError(
            f"No eval pairs for direction '{direction}'. "
            f"Available: {', '.join(sorted(EVAL_PAIRS))}"
        )
    return EVAL_PAIRS[direction]


def all_directions() -> list:
    """Return every direction key that has an eval set, e.g. ['es-en', ...]."""
    return list(EVAL_PAIRS.keys())


def eval_set_sizes() -> dict:
    """Return {direction: number_of_pairs} for quick reporting."""
    return {d: len(p) for d, p in EVAL_PAIRS.items()}
