import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "backend/emails.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# INTENT CLASSIFICATION MODEL
# ============================================================

classifier = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])


classifier.fit(
    df["text"],
    df["intent"]
)


# ============================================================
# SEARCH MODEL
# ============================================================

search_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

email_vectors = search_vectorizer.fit_transform(
    df["text"]
)


# ============================================================
# PRIORITY
# ============================================================

PRIORITY = {
    "Complaint": "High",
    "Refund": "High",
    "Cancellation": "High",
    "Order_Status": "Medium",
    "Account_Support": "Medium",
    "Product_Inquiry": "Low"
}


# ============================================================
# CLASSIFY EMAIL
# ============================================================

def classify_email(email):

    intent = classifier.predict([email])[0]

    probabilities = classifier.predict_proba([email])[0]

    confidence = max(probabilities)

    priority = PRIORITY.get(
        intent,
        "Medium"
    )

    return {
        "intent": intent,
        "confidence": float(confidence),
        "priority": priority
    }


# ============================================================
# SEARCH BACKEND EMAILS
# ============================================================

def search_emails(query, top_k=5):

    query_vector = search_vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        email_vectors
    )[0]

    ranked_indexes = similarities.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indexes:

        score = similarities[index]

        results.append({
            "email": df.iloc[index]["text"],
            "intent": df.iloc[index]["intent"],
            "similarity": float(score)
        })

    return results


# ============================================================
# GET DATASET STATISTICS
# ============================================================

def get_statistics():

    return {
        "total_emails": len(df),
        "total_intents": df["intent"].nunique(),
        "intent_counts": df["intent"].value_counts().to_dict()
    }