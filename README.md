# 3415wk10

# Model Comparisons -> Results from compare_models
                                Model  Accuracy  Precision    Recall  F1 Score
BERT Embeddings + Logistic Regression  0.892562   0.895480  0.893089  0.892435
         TF-IDF + Logistic Regression  0.851240   0.863915  0.851626  0.851657
CountVectorizer + Logistic Regression  0.834711   0.844753  0.835569  0.833920

# Required installations to run code:
pip install pandas scikit-learn gradio joblib sentence-transformers 