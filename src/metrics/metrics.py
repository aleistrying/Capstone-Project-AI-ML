"""
Core metric calculation functions for recommendation evaluation.
"""


def calculate_precision(predicted_ids, relevant_ids):
    """
    Precision = (# of relevant items recommended) / (# of items recommended)

    How many of the movies we recommended were actually relevant?

    Args:
        predicted_ids: List of movie IDs recommended by the system
        relevant_ids: List of movie IDs that are actually relevant to user

    Returns:
        float: Precision score (0.0 to 1.0)
    """
    if not predicted_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    predicted_set = set(predicted_ids)

    true_positives = len(relevant_set & predicted_set)

    return true_positives / len(predicted_set)


def calculate_recall(predicted_ids, relevant_ids):
    """
    Recall = (# of relevant items recommended) / (# of all relevant items)

    Of all the relevant movies, how many did we return?

    Args:
        predicted_ids: List of movie IDs recommended by the system
        relevant_ids: List of movie IDs that are actually relevant to user

    Returns:
        float: Recall score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    predicted_set = set(predicted_ids)

    true_positives = len(relevant_set & predicted_set)

    return true_positives / len(relevant_set)


def calculate_accuracy(predicted_ids, relevant_ids, total_items):
    """
    Accuracy = (# of correct predictions) / (# of total items)

    For recommendation systems, accuracy measures what fraction of items
    were correctly classified as relevant or not relevant.

    Args:
        predicted_ids: List of movie IDs recommended by the system
        relevant_ids: List of movie IDs that are actually relevant
        total_items: Total number of items in the dataset

    Returns:
        float: Accuracy score (0.0 to 1.0)
    """
    if total_items == 0:
        return 0.0

    relevant_set = set(relevant_ids)
    predicted_set = set(predicted_ids)

    # True Positives: recommended AND relevant
    true_positives = len(relevant_set & predicted_set)

    # True Negatives: not recommended AND not relevant
    all_items = set(range(total_items))
    not_relevant = all_items - relevant_set
    not_predicted = all_items - predicted_set
    true_negatives = len(not_relevant & not_predicted)

    return (true_positives + true_negatives) / total_items


def calculate_f1_score(predicted_ids, relevant_ids):
    """
    F1 Score = 2 * (precision * recall) / (precision + recall)

    Harmonic mean of precision and recall. Good for imbalanced data.

    Args:
        predicted_ids: List of movie IDs recommended by the system
        relevant_ids: List of movie IDs that are actually relevant

    Returns:
        float: F1 score (0.0 to 1.0)
    """
    precision = calculate_precision(predicted_ids, relevant_ids)
    recall = calculate_recall(predicted_ids, relevant_ids)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def calculate_mean_reciprocal_rank(predicted_ids, relevant_ids):
    """
    MRR = 1 / (rank of first relevant item)

    Measures how early the first relevant item appears in the ranking.

    Args:
        predicted_ids: Ordered list of movie IDs recommended by the system
        relevant_ids: List of movie IDs that are actually relevant

    Returns:
        float: MRR score (0.0 to 1.0)
    """
    relevant_set = set(relevant_ids)

    for rank, movie_id in enumerate(predicted_ids, start=1):
        if movie_id in relevant_set:
            return 1.0 / rank

    return 0.0
