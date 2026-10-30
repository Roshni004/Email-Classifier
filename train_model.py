
import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. Project paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR, "dataset", "emails.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# 2. Load dataset
# ==========================================

data = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Total emails:", len(data))

# Validate required columns
required_columns = [
    "text",
    "spam",
    "category",
    "priority"
]

for column in required_columns:
    if column not in data.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

# ==========================================
# 3. Data cleaning
# ==========================================

data = data.dropna(
    subset=["text", "spam", "category", "priority"]
).copy()

data["text"] = data["text"].astype(str).str.lower()

data["spam"] = data["spam"].astype(int)

print("Missing values removed.")
print("Cleaned emails:", len(data))

# ==========================================
# 4. Exploratory Data Analysis summary
# ==========================================

dataset_summary = {
    "total_emails": int(len(data)),
    "spam_count": int((data["spam"] == 1).sum()),
    "not_spam_count": int((data["spam"] == 0).sum()),
    "categories": {
        str(k): int(v)
        for k, v in data["category"].value_counts().items()
    },
    "priorities": {
        str(k): int(v)
        for k, v in data["priority"].value_counts().items()
    },
    "average_text_length": round(
        float(data["text"].str.len().mean()), 2
    )
}

with open(
    os.path.join(BASE_DIR, "dataset_summary.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(dataset_summary, file, indent=4)

print("\nEDA summary saved.")

# ==========================================
# 5. TF-IDF vectorization
# ==========================================

X = data["text"]

y_spam = data["spam"]
y_category = data["category"]
y_priority = data["priority"]

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_tfidf = vectorizer.fit_transform(X)

print("\nTF-IDF transformation completed.")
print("Feature matrix shape:", X_tfidf.shape)

# ==========================================
# 6. Evaluation helper
# ==========================================

def evaluate_model(model, X_test, y_test, name):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    ).tolist()

    print(f"\n========== {name} ==========")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": matrix,
        "classification_report": report
    }

# ==========================================
# 7. Train spam model
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y_spam,
    test_size=0.2,
    random_state=42,
    stratify=y_spam
)

spam_model = LogisticRegression(max_iter=1000)

spam_model.fit(X_train, y_train)

spam_metrics = evaluate_model(
    spam_model,
    X_test,
    y_test,
    "SPAM MODEL"
)

# ==========================================
# 8. Train category model
# ==========================================

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    X_tfidf,
    y_category,
    test_size=0.2,
    random_state=42,
    stratify=y_category
)

category_model = LogisticRegression(max_iter=1000)

category_model.fit(X_train_cat, y_train_cat)

category_metrics = evaluate_model(
    category_model,
    X_test_cat,
    y_test_cat,
    "CATEGORY MODEL"
)

# ==========================================
# 9. Train priority model
# ==========================================

X_train_pri, X_test_pri, y_train_pri, y_test_pri = train_test_split(
    X_tfidf,
    y_priority,
    test_size=0.2,
    random_state=42,
    stratify=y_priority
)

priority_model = LogisticRegression(max_iter=1000)

priority_model.fit(X_train_pri, y_train_pri)

priority_metrics = evaluate_model(
    priority_model,
    X_test_pri,
    y_test_pri,
    "PRIORITY MODEL"
)

# ==========================================
# 10. Save model metrics
# ==========================================

all_metrics = {
    "spam": spam_metrics,
    "category": category_metrics,
    "priority": priority_metrics
}

with open(
    os.path.join(BASE_DIR, "model_metrics.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(all_metrics, file, indent=4)

print("\nModel metrics saved.")

# ==========================================
# 11. Save important spam words
# ==========================================

feature_names = vectorizer.get_feature_names_out()

spam_coefficients = spam_model.coef_[0]

top_spam_indices = spam_coefficients.argsort()[-20:][::-1]

important_spam_words = [
    {
        "word": str(feature_names[index]),
        "weight": round(
            float(spam_coefficients[index]), 4
        )
    }
    for index in top_spam_indices
]

with open(
    os.path.join(BASE_DIR, "spam_keywords.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(important_spam_words, file, indent=4)

print("Spam keyword analysis saved.")

# ==========================================
# 12. Save trained models
# ==========================================

joblib.dump(
    vectorizer,
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
)

joblib.dump(
    spam_model,
    os.path.join(MODEL_DIR, "spam_model.pkl")
)

joblib.dump(
    category_model,
    os.path.join(MODEL_DIR, "category_model.pkl")
)

joblib.dump(
    priority_model,
    os.path.join(MODEL_DIR, "priority_model.pkl")
)

print("\n================================")
print("All models saved successfully!")
print("================================")