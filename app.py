import re
from typing import List

import gradio as gr
import joblib
from sentence_transformers import SentenceTransformer

MODEL_FILE = "best_classifier.joblib"
BERT_MODEL_NAME = "all-MiniLM-L6-v2"

ISSUE_RULES = {
    "Product Defect": {
        "keywords": [
            "broken", "damaged", "faulty", "defective", "cracked",
            "not working", "stopped working", "leaking", "scratched", "malfunction"
        ],
        "department": "Quality Assurance / Returns Team"
    },
    "Delivery Problem": {
        "keywords": [
            "late", "delayed", "never arrived", "missing",
            "wrong item", "wrong product", "not delivered", "shipment delay"
        ],
        "department": "Logistics / Fulfilment Team"
    },
    "Customer Service Problem": {
        "keywords": [
            "refund", "customer service", "no response",
            "ignored", "rude", "unhelpful", "terrible support", "no reply"
        ],
        "department": "Customer Service Team"
    }
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_issues(text):
    cleaned = clean_text(text)
    found_issues = []
    departments = []

    for issue_type, info in ISSUE_RULES.items():
        if any(keyword in cleaned for keyword in info["keywords"]):
            found_issues.append(issue_type)
            departments.append(info["department"])

    return found_issues, sorted(set(departments))

def build_priority(sentiment, issues):
    if sentiment == "negative" and issues:
        return "High Priority"
    if sentiment == "negative":
        return "Medium Priority"
    if sentiment == "neutral" and issues:
        return "Review Needed"
    return "No Immediate Action"

def format_list(items, empty_text):
    if not items:
        return empty_text
    return ", ".join(items)

# Load saved classifier and BERT encoder
MODEL = joblib.load(MODEL_FILE)
BERT_ENCODER = SentenceTransformer(BERT_MODEL_NAME)

def predict_review(user_text):
    if not user_text or not user_text.strip():
        return "Please enter a review.", "No issue detected", "No escalation needed", "No flag"

    cleaned = clean_text(user_text)
    emb = BERT_ENCODER.encode([cleaned])
    sentiment = MODEL.predict(emb)[0]

    issues, departments = detect_issues(user_text)
    priority = build_priority(sentiment, issues)

    return (
        sentiment,
        format_list(issues, "No issue detected"),
        format_list(departments, "No escalation needed"),
        priority
    )

demo = gr.Interface(
    fn=predict_review,
    inputs=gr.Textbox(
        lines=6,
        label="Enter customer review",
        placeholder="Example: The item arrived damaged and I want a refund."
    ),
    outputs=[
        gr.Textbox(label="Predicted sentiment"),
        gr.Textbox(label="Issue type"),
        gr.Textbox(label="Escalate to department"),
        gr.Textbox(label="Priority flag")
    ],
    title="Retail Review Sentiment & Escalation Checker",
    description="Enter a review to predict sentiment, detect issue type, and route it to the correct department.",
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()