
from pathlib import Path
import json
import re

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
pinned_emails = set()
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # Maximum upload size: 10 MB

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "emails.csv"
MODEL_DIR = BASE_DIR / "models"

# --------------------------------------------------
# Load trained models
# --------------------------------------------------
def load_model(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model: {path}. Run python train_model.py first."
        )
    return joblib.load(path)


spam_model = load_model("spam_model.pkl")
category_model = load_model("category_model.pkl")
priority_model = load_model("priority_model.pkl")
vectorizer = load_model("tfidf_vectorizer.pkl")


# --------------------------------------------------
# Load and prepare dataset
# --------------------------------------------------
def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required = {"text", "spam", "category", "priority"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "emails.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    df["text"] = df["text"].fillna("").astype(str)
    df["category"] = df["category"].fillna("Unknown").astype(str)
    df["priority"] = df["priority"].fillna("Unknown").astype(str)

    return df


def normalize_spam(value):
    value = str(value).strip().lower()

    if value in {"1", "1.0", "true", "spam"}:
        return "Spam"

    if value in {"0", "0.0", "false", "ham", "not spam"}:
        return "Not Spam"

    return "Unknown"


def json_ready(value):
    """Convert NumPy/Pandas values into JSON-safe Python values."""
    if hasattr(value, "item"):
        return value.item()
    return value


# --------------------------------------------------
# Exploratory Data Analysis
# --------------------------------------------------
def calculate_eda(df):
    text = df["text"].fillna("").astype(str).str.strip()
    nonempty = text[text != ""]
    lengths = nonempty.str.split().str.len()

    spam_labels = df["spam"].map(normalize_spam)
    categories = df["category"].replace("", "Unknown")
    priorities = df["priority"].replace("", "Unknown")

    # Detect duplicate non-empty email text, ignoring case/spacing.
    normalized_text = (
        text.str.lower().str.replace(r"\s+", " ", regex=True)
    )
    duplicates = int(
        normalized_text[text != ""].duplicated().sum()
    )

    # Frequent words after basic stopword removal.
    stopwords = {
        "the", "and", "for", "you", "are", "your", "this",
        "that", "with", "from", "have", "has", "was", "will",
        "our", "not", "but", "can", "all", "please", "dear",
        "subject", "hello", "hi", "to", "of", "in", "is", "a"
    }

    word_counts = {}
    for message in nonempty:
        words = re.findall(r"\b[a-zA-Z]{2,}\b", message.lower())
        for word in words:
            if word not in stopwords:
                word_counts[word] = word_counts.get(word, 0) + 1

    top_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )[:10]

    # Word count ranges.
    length_labels = ["1–5", "6–10", "11–20", "21–40", "41+"]
    length_values = [
        int(((lengths >= 1) & (lengths <= 5)).sum()),
        int(((lengths >= 6) & (lengths <= 10)).sum()),
        int(((lengths >= 11) & (lengths <= 20)).sum()),
        int(((lengths >= 21) & (lengths <= 40)).sum()),
        int((lengths >= 41).sum())
    ]

    # Spam counts, including unknown labels if present.
    spam_counts = spam_labels.value_counts()
    category_counts = categories.value_counts()
    priority_counts = priorities.value_counts()

    # Category vs spam cross-analysis.
    cross = pd.crosstab(categories, spam_labels)

    return {
        "total": int(len(df)),
        "nonempty": int((text != "").sum()),
        "empty": int((text == "").sum()),
        "missing": int(df[["text", "spam", "category", "priority"]]
                        .isna().sum().sum()),
        "missing_by_column": {
            col: int(df[col].isna().sum())
            for col in ["text", "spam", "category", "priority"]
        },
        "duplicates": duplicates,
        "average_words": round(float(lengths.mean()), 2) if len(lengths) else 0,
        "median_words": round(float(lengths.median()), 2) if len(lengths) else 0,
        "max_words": int(lengths.max()) if len(lengths) else 0,
        "spam_labels": spam_counts.index.tolist(),
        "spam_values": [int(v) for v in spam_counts.values],
        "category_labels": category_counts.index.tolist(),
        "category_values": [int(v) for v in category_counts.values],
        "priority_labels": priority_counts.index.tolist(),
        "priority_values": [int(v) for v in priority_counts.values],
        "length_labels": length_labels,
        "length_values": length_values,
        "word_labels": [word for word, _ in top_words],
        "word_values": [int(count) for _, count in top_words],
        "cross_labels": cross.index.tolist(),
        "cross_spam": [
            int(cross.loc[label].get("Spam", 0))
            for label in cross.index
        ],
        "cross_legitimate": [
            int(cross.loc[label].get("Not Spam", 0))
            for label in cross.index
        ]
    }


# --------------------------------------------------
# Load optional model evaluation metrics
# --------------------------------------------------
def load_metrics():
    path = BASE_DIR / "model_metrics.json"

    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        app.logger.exception("Could not read model_metrics.json")
        return {}



# --------------------------------------------------
# Dashboard
# --------------------------------------------------
@app.route("/")
def dashboard():
    df = load_dataset()
    eda = calculate_eda(df)

    # Show actual dataset records; these are labels, not fresh predictions.
    emails = []
    for index, row in df.head(50).iterrows():
        emails.append({
            "id": int(index) + 1,
            "text": row["text"],
            "spam": normalize_spam(row["spam"]),
            "category": row["category"],
            "priority": row["priority"]
        })

    # Put pinned emails first in the table
    emails.sort(
        key=lambda email: str(email["id"]) not in pinned_emails
    )

    return render_template(
        "index.html",
        eda=eda,
        emails=emails,
        pinned_emails=pinned_emails,
        metrics=load_metrics(),
        result=None,
        error=None
    )


# --------------------------------------------------
# Analyze one email using the existing trained models
# --------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze_email():
    df = load_dataset()
    eda = calculate_eda(df)

    email_text = request.form.get("email_text", "").strip()

    result = None
    error = None

    if not email_text:
        error = "Please enter an email message before analyzing."
    else:
        try:
            features = vectorizer.transform([email_text])

            spam_prediction = spam_model.predict(features)[0]
            category_prediction = category_model.predict(features)[0]
            priority_prediction = priority_model.predict(features)[0]

            spam_status = normalize_spam(spam_prediction)

            result = {
                "text": email_text,
                "spam": spam_status,
                "category": str(category_prediction),
                "priority": str(priority_prediction)
            }

            # Confidence is only shown if the model supports probabilities.
            if hasattr(spam_model, "predict_proba"):
                probabilities = spam_model.predict_proba(features)[0]
                result["confidence"] = round(
                    float(max(probabilities)) * 100, 1
                )

        except Exception:
            app.logger.exception("Email classification failed")
            error = (
                "Prediction failed. Check that your trained models "
                "and TF-IDF vectorizer are compatible."
            )

    emails = [
        {
            "id": int(index) + 1,
            "text": row["text"],
            "spam": normalize_spam(row["spam"]),
            "category": row["category"],
            "priority": row["priority"]
        }
        for index, row in df.head(50).iterrows()
    ]

    return render_template(
        "index.html",
        eda=eda,
        emails=emails,
        pinned_emails=pinned_emails,
        metrics=load_metrics(),
        result=result,
        error=error
    )

# --------------------------------------------------
# Upload and analyze a CSV file
# --------------------------------------------------
@app.route("/upload-csv", methods=["POST"])
def upload_csv():
    file = request.files.get("csv_file")

    if file is None or not file.filename:
        return "Please select a CSV file.", 400

    if not file.filename.lower().endswith(".csv"):
        return "Please upload a CSV file only.", 400

    try:
        df = pd.read_csv(file, nrows=50000, encoding="utf-8-sig")

        if df.empty:
            return "The CSV file is empty.", 400

        # Check required columns for your existing EDA
        required = {"text", "spam", "category", "priority"}
        missing = required - set(df.columns)

        if missing:
            return (
                "Your CSV is missing these required columns: "
                + ", ".join(sorted(missing))
            ), 400

        # Clean uploaded data
        df["text"] = df["text"].fillna("").astype(str)
        df["category"] = (
            df["category"].fillna("Unknown").astype(str)
        )
        df["priority"] = (
            df["priority"].fillna("Unknown").astype(str)
        )

        # Calculate EDA using the uploaded dataset
        eda = calculate_eda(df)

        # Prepare records for the dashboard table
        emails = [
            {
                "id": int(index) + 1,
                "text": row["text"],
                "spam": normalize_spam(row["spam"]),
                "category": row["category"],
                "priority": row["priority"]
            }
            for index, row in df.head(50).iterrows()
        ]

        return render_template(
            "index.html",
            eda=eda,
            emails=emails,
            pinned_emails=pinned_emails,
            metrics=load_metrics(),
            result=None,
            error=None
        )

    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        return f"Unable to analyze this CSV: {exc}", 400


# --------------------------------------------------
# Pin / Unpin emails
# --------------------------------------------------
@app.route("/toggle-pin", methods=["POST"])
def toggle_pin():
    email_id = request.form.get("email_id", "").strip()

    if not email_id.isdigit():
        return "Invalid email ID", 400

    if email_id in pinned_emails:
        pinned_emails.remove(email_id)
    else:
        pinned_emails.add(email_id)

    return redirect(url_for("dashboard"))



if __name__ == "__main__":
    app.run(debug=True)
