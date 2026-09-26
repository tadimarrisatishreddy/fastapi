"""
Evaluate a saved spam detection model.
Reloads spam_model.joblib + vectorizer.joblib, re-splits the dataset the
same way training did, and prints accuracy/precision/recall/F1.
"""

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Reuse the same cleaning + loading logic as the main script
from spamdetection import clean_text, load_data, DATA_PATH, LABEL_COL, TEXT_COL

MODEL_PATH = "spam_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

# Load saved model + vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Rebuild the same test split used during training (random_state=42 matches)
df = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label_num"],
    test_size=0.2, random_state=42, stratify=df["label_num"]
)

X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, y_pred))
print("\n" + classification_report(y_test, y_pred, target_names=["ham", "spam"]))