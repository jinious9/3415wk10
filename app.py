import re
import warnings
from io import BytesIO
from typing import Dict, List, Tuple

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# Replace this with your RAW GitHub CSV link.
# Example:
# https://raw.githubusercontent.com/your-username/your-repo/main/sample.csv
# ============================================================
CSV_URL = "PASTE_YOUR_RAW_GITHUB_CSV_LINK_HERE"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_PER_CLASS = 2000  # keeps training fast for demo and assignment
MAX_FEATURES = 12000
OUTPUT_MODEL_PATH = "best_sentiment_model.joblib"
OUTPUT_LABELS_PATH = "model_labels.joblib"

TEXT_CANDIDATES = ["text", "Text", "review", "Review", "reviews", "summary", "Summary"]
LABEL_CANDIDATES = ["Sentiment", "sentiment", "label", "Label"]
SUMMARY_CANDIDATES = ["Summary", "summary"]
REVIEW_CANDIDATES = ["Review", "review"]

ISSUE_PATTERNS = {
    "defect_issue": [
        "broken", "damaged", "faulty", "defective", "cracked", "not working",
        "stopped working", "poor quality", "leaking", "scratched", "malfunction"
    ],
    "delivery_issue": [
        "late", "delayed", "never arrived", "missing", "wrong item", "wrong product",
        "delivered late", "not delivered", "package missing", "shipment delay"
    ],
    "service_issue": [
        "refund", "customer service", "no response", "ignored", "hung up", "rude",
        "no reply", "unhelpful", "terrible support"
    ],
}

# ============================================================
# HELPERS
# ============================================================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_first_existing(columns: List[str], candidates: List[str]):
    for c in candidates:
        if c in columns:
            return c
    return None


def load_dataset(csv_url: str) -> pd.DataFrame:
    if not csv_url or "PASTE_YOUR_RAW_GITHUB_CSV_LINK_HERE" in csv_url:
        raise ValueError("Replace CSV_URL with your raw GitHub CSV link before running the app.")

    response = requests.get(csv_url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content))


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    label_col = find_first_existing(cols, LABEL_CANDIDATES)
    if label_col is None:
        raise ValueError(f"Could not find a label column. Found columns: {cols}")

    summary_col = find_first_existing(cols, SUMMARY_CANDIDATES)
    review_col = find_first_existing(cols, REVIEW_CANDIDATES)
    text_col = find_first_existing(cols, TEXT_CANDIDATES)

    work_df = df.copy()

    if summary_col and review_col:
        work_df["text"] = (
            work_df[summary_col].fillna("").astype(str) + " " + work_df[review_col].fillna("").astype(str)
        ).str.strip()
    elif text_col is not None:
        work_df["text"] = work_df[text_col].fillna("").astype(str)
    elif review_col is not None:
        work_df["text"] = work_df[review_col].fillna("").astype(str)
    elif summary_col is not None:
        work_df["text"] = work_df[summary_col].fillna("").astype(str)
    else:
        raise ValueError(f"Could not find a usable text column. Found columns: {cols}")

    work_df["label"] = work_df[label_col].astype(str).str.strip().str.lower()
    work_df["text"] = work_df["text"].apply(clean_text)
    work_df = work_df[["text", "label"]].dropna()
    work_df = work_df[work_df["text"].str.len() > 0].copy()

    valid_labels = {"positive", "negative", "neutral"}
    if work_df["label"].isin(valid_labels).sum() == 0:
        raise ValueError("Label column found, but it does not seem to contain sentiment labels like positive/negative/neutral.")

    work_df = work_df[work_df["label"].isin(valid_labels)].copy()
    return work_df


def stratified_sample(df: pd.DataFrame, max_per_class: int = MAX_PER_CLASS) -> pd.DataFrame:
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), max_per_class), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )


def detect_issue_types(text: str) -> List[str]:
    cleaned = clean_text(text)
    found = []
    for issue_type, patterns in ISSUE_PATTERNS.items():
        if any(pattern in cleaned for pattern in patterns):
            found.append(issue_type)
    return found


def build_business_flag(predicted_label: str, issue_types: List[str]) -> str:
    if predicted_label == "negative" and issue_types:
        return "Flagged: high-priority problematic order"
    if predicted_label == "negative":
        return "Flagged: negative review requires follow-up"
    if predicted_label == "neutral" and issue_types:
        return "Review recommended: possible order issue"
    return "Not flagged"


def explain_flag(issue_types: List[str]) -> str:
    if not issue_types:
        return "No issue keywords detected."
    readable = ", ".join(issue_types)
    return f"Detected issue categories: {readable}."


def get_candidate_models() -> Dict[str, Pipeline]:
    models = {
        "CountVectorizer + LogisticRegression": Pipeline([
            ("vectorizer", CountVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1200))
        ]),
        "TFIDF + LogisticRegression": Pipeline([
            ("vectorizer", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1200))
        ])
    }

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.preprocessing import FunctionTransformer

        bert_encoder = SentenceTransformer("all-MiniLM-L6-v2")

        def encode_texts(texts):
            texts = list(texts)
            return bert_encoder.encode(texts, show_progress_bar=False)

        bert_pipeline = Pipeline([
            ("encoder", FunctionTransformer(encode_texts, validate=False)),
            ("classifier", LogisticRegression(max_iter=1200))
        ])
        models["MiniLM Embeddings + LogisticRegression"] = bert_pipeline
    except Exception:
        pass

    return models


def evaluate_models(df: pd.DataFrame):
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    candidate_models = get_candidate_models()
    results = []
    fitted_models = {}

    for name, model in candidate_models.items():
        working_model = clone(model)
        working_model.fit(X_train, y_train)
        preds = working_model.predict(X_test)

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, preds),
            "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
        }
        results.append(metrics)
        fitted_models[name] = {
            "model": working_model,
            "preds": preds,
            "y_test": y_test,
            "report": classification_report(y_test, preds, zero_division=0),
            "cm": confusion_matrix(y_test, preds, labels=sorted(y.unique())),
        }

    results_df = pd.DataFrame(results).sort_values(["f1_macro", "accuracy"], ascending=False).reset_index(drop=True)
    best_name = results_df.iloc[0]["model"]
    best_bundle = fitted_models[best_name]

    return {
        "results_df": results_df,
        "best_model_name": best_name,
        "best_model": best_bundle["model"],
        "best_report": best_bundle["report"],
        "best_cm": best_bundle["cm"],
        "labels": sorted(y.unique()),
        "X_train_size": len(X_train),
        "X_test_size": len(X_test),
        "fitted_models": fitted_models,
    }


def plot_confusion_matrix(cm: np.ndarray, labels: List[str]):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def get_top_features(trained_model, top_n: int = 10) -> pd.DataFrame:
    try:
        vectorizer = trained_model.named_steps.get("vectorizer")
        classifier = trained_model.named_steps.get("classifier")
        if vectorizer is None or classifier is None:
            return pd.DataFrame({"info": ["Top features not available for this model type."]})

        feature_names = np.array(vectorizer.get_feature_names_out())
        classes = classifier.classes_
        coef = classifier.coef_
        rows = []

        for idx, class_name in enumerate(classes):
            top_idx = np.argsort(coef[idx])[-top_n:][::-1]
            for rank, feature_idx in enumerate(top_idx, start=1):
                rows.append({
                    "class": class_name,
                    "rank": rank,
                    "feature": feature_names[feature_idx],
                    "weight": round(float(coef[idx][feature_idx]), 4)
                })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame({"info": ["Top features not available for this model type."]})


def predict_review(user_text: str):
    if not user_text or not user_text.strip():
        return (
            "Please enter a review.",
            pd.DataFrame(),
            "No flag yet.",
            "No explanation yet."
        )

    cleaned = clean_text(user_text)
    pred = BEST_MODEL.predict([cleaned])[0]
    issue_types = detect_issue_types(user_text)
    business_flag = build_business_flag(pred, issue_types)
    flag_explanation = explain_flag(issue_types)

    try:
        probs = BEST_MODEL.predict_proba([cleaned])[0]
        classes = BEST_MODEL.named_steps["classifier"].classes_
        prob_df = pd.DataFrame({
            "label": classes,
            "probability": np.round(probs, 4)
        }).sort_values("probability", ascending=False)
    except Exception:
        prob_df = pd.DataFrame({"info": ["Probability output not available for this model type."]})

    return pred, prob_df, business_flag, flag_explanation


def save_artifacts(model, labels: List[str]):
    joblib.dump(model, OUTPUT_MODEL_PATH)
    joblib.dump(labels, OUTPUT_LABELS_PATH)


# ============================================================
# TRAIN ON STARTUP
# ============================================================
print("Loading dataset from GitHub...")
raw_df = load_dataset(CSV_URL)
processed_df = prepare_dataset(raw_df)
sampled_df = stratified_sample(processed_df, max_per_class=MAX_PER_CLASS)

print("Training and comparing models...")
evaluation_bundle = evaluate_models(sampled_df)
BEST_MODEL = evaluation_bundle["best_model"]
BEST_MODEL_NAME = evaluation_bundle["best_model_name"]
RESULTS_DF = evaluation_bundle["results_df"]
BEST_REPORT = evaluation_bundle["best_report"]
BEST_CM = evaluation_bundle["best_cm"]
CLASS_LABELS = evaluation_bundle["labels"]
TOP_FEATURES_DF = get_top_features(BEST_MODEL)
CM_FIG = plot_confusion_matrix(BEST_CM, CLASS_LABELS)
save_artifacts(BEST_MODEL, CLASS_LABELS)

print("Best model:", BEST_MODEL_NAME)
print(RESULTS_DF)
print(BEST_REPORT)


# ============================================================
# GRADIO APP
# ============================================================
summary_markdown = f"""
# Retail Review Intelligence App

This app supports the assignment problem of identifying problematic customer orders from retail review text.

### What the app does
1. Classifies review sentiment into **positive**, **neutral**, or **negative**.
2. Detects likely issue types such as **defect**, **delivery**, or **service** problems.
3. Applies business logic to flag reviews that may require follow-up.
4. Compares multiple models and shows evaluation metrics.

### Dataset used in training
- Total rows after sampling: **{len(sampled_df)}**
- Train size: **{evaluation_bundle['X_train_size']}**
- Test size: **{evaluation_bundle['X_test_size']}**
- Best model selected by macro F1: **{BEST_MODEL_NAME}**

### Suggested demo inputs
- The item arrived damaged and I want a refund.
- Good quality product and fast delivery.
- Packaging was okay but the wrong item was sent.
"""

with gr.Blocks() as demo:
    gr.Markdown(summary_markdown)

    with gr.Tab("Live Prediction"):
        review_input = gr.Textbox(
            lines=6,
            label="Enter a customer review",
            placeholder="Example: The item arrived damaged and customer service ignored my refund request."
        )
        predict_btn = gr.Button("Predict")
        pred_output = gr.Textbox(label="Predicted sentiment")
        prob_output = gr.Dataframe(label="Class probabilities")
        flag_output = gr.Textbox(label="Business flag")
        explain_output = gr.Textbox(label="Flag explanation")

        predict_btn.click(
            fn=predict_review,
            inputs=review_input,
            outputs=[pred_output, prob_output, flag_output, explain_output]
        )

    with gr.Tab("Model Comparison"):
        gr.Markdown("## Model performance comparison")
        comparison_df = gr.Dataframe(value=RESULTS_DF, label="Evaluation metrics")
        gr.Markdown("Higher macro F1 is generally better when comparing balanced multi-class performance.")

    with gr.Tab("Confusion Matrix"):
        gr.Markdown("## Best model confusion matrix")
        gr.Plot(value=CM_FIG)

    with gr.Tab("Explainability"):
        gr.Markdown("## Top indicative features for the best linear model")
        gr.Dataframe(value=TOP_FEATURES_DF, label="Top features")

    with gr.Tab("Classification Report"):
        gr.Code(value=BEST_REPORT, language="text", label="Detailed classification report")

    with gr.Tab("Assignment Notes"):
        gr.Markdown(
            """
## How to talk about this in your video
- The business problem is that retailers receive too many reviews to inspect manually.
- The first layer predicts sentiment from review text.
- The second layer detects likely issue categories from keywords and phrases.
- The final flagging logic helps prioritise reviews for follow-up.
- The app compares multiple text-classification approaches and selects the best one by macro F1.
- The output is interpretable because the business flag and issue categories are transparent.

## Why this is stronger for the assignment
- It includes preprocessing, model training, evaluation, explainability, and deployment.
- It combines sentiment prediction with business rules to flag problematic orders.
- It is reusable because the best model is saved as a joblib artifact.
            """
        )

if __name__ == "__main__":
    demo.launch()
