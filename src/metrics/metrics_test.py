"""Experimental lexical-overlap accuracy metric.

This module is retained for backward compatibility. New evaluation code should
prefer the metrics exposed by :mod:`src.metrics.metrics`.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity


def _binary_overlap_vectors(recommendations, query_words, vocabulary, vectors):
    """Return expected and predicted lexical-overlap labels."""
    query_indices = [vocabulary[word] for word in query_words]
    expected = [
        int(all(word in overview.lower() for word in query_words))
        for overview in recommendations["overview"]
    ]
    predicted = [
        int(all(vector[index] for index in query_indices)) for vector in vectors
    ]
    return expected, predicted


def accuracy(recommendations, user_query):
    """Calculate lexical classification scores and average cosine similarity."""
    vectorizer = CountVectorizer(binary=True)
    texts = [*recommendations["overview"].tolist(), user_query]
    feature_matrix = vectorizer.fit_transform(texts)
    query_vector = feature_matrix[-1].toarray()[0]
    overview_vectors = feature_matrix[:-1].toarray()
    query_words = [
        word for word in user_query.lower().split() if word in vectorizer.vocabulary_
    ]
    expected, predicted = _binary_overlap_vectors(
        recommendations,
        query_words,
        vectorizer.vocabulary_,
        overview_vectors,
    )
    similarities = cosine_similarity([query_vector], overview_vectors)[0]
    return {
        "accuracy": accuracy_score(expected, predicted),
        "precision": precision_score(expected, predicted, zero_division=0),
        "recall": recall_score(expected, predicted, zero_division=0),
        "similarity": float(np.mean(similarities)),
    }
