"""
Spam Detection Model
=====================
A complete pipeline: load data -> clean text -> vectorize (TF-IDF) ->
train a classifier -> evaluate -> save model -> predict on new messages.

Dataset: SMS Spam Collection (public dataset, ~5,500 labeled texts).
Download it here if you don't have it locally:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Expected format: a CSV with two columns -> label ('spam'/'ham'), text
If your CSV has different column names, adjust LABEL_COL / TEXT_COL below.
"""

import re
import string
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ---------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------
DATA_PATH = r"C:\Users\HP 440 G8\Downloads\sms+spam+collection (1)\SMSSpamCollection" # path to your dataset
LABEL_COL = "label"              # column with 'spam' / 'ham'
TEXT_COL = "text"                # column with the message text
MODEL_OUT = "spam_model.joblib"
VECTORIZER_OUT = "vectorizer.joblib"
ALGORITHM = "logistic_regression"        # or "logistic_regression"


# ---------------------------------------------------------------------
# 2. TEXT CLEANING
# ---------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, strip URLs/numbers/punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"\d+", " ", text)                        # numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------
# 3. LOAD + PREPARE DATA
# ---------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=[LABEL_COL, TEXT_COL], encoding="latin-1")
    df = df[[LABEL_COL, TEXT_COL]].dropna()
    df["clean_text"] = df[TEXT_COL].apply(clean_text)
    df["label_num"] = df[LABEL_COL].map({"ham": 0, "spam": 1})
    return df


# ---------------------------------------------------------------------
# 4. TRAIN
# ---------------------------------------------------------------------
def train_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label_num"],
        test_size=0.2, random_state=42, stratify=df["label_num"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)   # unigrams + bigrams catch phrases like "click here"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    if ALGORITHM == "logistic_regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        model = MultinomialNB()

    model.fit(X_train_vec, y_train)

    # ---- Evaluation ----
    y_pred = model.predict(X_test_vec)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print("\n" + classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    return model, vectorizer


# ---------------------------------------------------------------------
# 5. PREDICT ON NEW MESSAGES
# ---------------------------------------------------------------------
def predict(messages, model, vectorizer):
    cleaned = [clean_text(m) for m in messages]
    vectors = vectorizer.transform(cleaned)
    preds = model.predict(vectors)
    probs = model.predict_proba(vectors)[:, 1]  # probability of spam
    return [
        {"message": m, "prediction": "SPAM" if p == 1 else "HAM", "spam_probability": round(float(prob), 3)}
        for m, p, prob in zip(messages, preds, probs)
    ]


# ---------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    model, vectorizer = train_model(df)

    # Save for later reuse without retraining
    joblib.dump(model, MODEL_OUT)
    joblib.dump(vectorizer, VECTORIZER_OUT)
    print(f"\nModel saved to {MODEL_OUT}, vectorizer saved to {VECTORIZER_OUT}")

    # Try it on new, unseen messages
    test_messages = [
        "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!",
        "Hey, are we still on for lunch tomorrow?",
        "URGENT: Your account will be suspended. Verify your details immediately.",
    ]
    results = predict(test_messages, model, vectorizer)
    for r in results:
        print(r)