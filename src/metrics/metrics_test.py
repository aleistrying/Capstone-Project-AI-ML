def accuracy(recommendations, user_query):
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Prepare ground truth and predictions
    vectorizer = CountVectorizer(binary=True)
    texts = recommendations['overview'].tolist()
    texts.append(user_query)
    X = vectorizer.fit_transform(texts)
    query_vec = X[-1].toarray()[0]
    overviews_vec = X[:-1].toarray()

    # Get indices of words in user_query
    query_words = [w for w in user_query.lower().split() if w in vectorizer.vocabulary_]
    query_indices = [vectorizer.vocabulary_[w] for w in query_words]

    # For each overview, check if it matches the query (simple overlap)
    y_true = [1 if all(w in o.lower() for w in query_words) else 0 for o in recommendations['overview']]
    y_pred = []
    for vec in overviews_vec:
        if all(vec[idx] for idx in query_indices):
            y_pred.append(1)
        else:
            y_pred.append(0)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    # Calculate cosine similarity between user_query and each overview
    similarities = cosine_similarity([query_vec], overviews_vec)[0]
    avg_similarity = float(np.mean(similarities))

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'similarity': avg_similarity}