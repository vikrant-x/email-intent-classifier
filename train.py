import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("emails.csv")

X = df["text"]
y = df["intent"]


# ==========================================
# 2. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. CREATE NLP PIPELINE
# ==========================================

model = Pipeline([
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
        LogisticRegression(max_iter=1000)
    )
])


# ==========================================
# 4. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 5. EVALUATE MODEL
# ==========================================

predictions = model.predict(X_test)

print("\n===== MODEL EVALUATION =====\n")

print(classification_report(
    y_test,
    predictions
))


# ==========================================
# 6. TEST EMAILS
# ==========================================

test_emails = [
    "I want my money back",
    "Where is my package?",
    "I forgot my password",
    "I want to cancel my order"
]


print("\n===== TEST PREDICTIONS =====\n")


for email in test_emails:

    # Predict intent
    prediction = model.predict([email])[0]

    # Get probabilities
    probabilities = model.predict_proba([email])[0]

    # Get highest probability
    confidence = max(probabilities)

    print("Email:", email)
    print("Predicted Intent:", prediction)
    print(f"Confidence: {confidence:.2%}")
    print()