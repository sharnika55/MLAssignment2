import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# -----------------------------
# Safe paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "breast_cancer.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"].map({"M": 1, "B": 0})

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Load scaler
# -----------------------------
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Evaluate models
# -----------------------------
models = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]

results = []

for name in models:
    model = joblib.load(os.path.join(MODEL_DIR, f"{name}.pkl"))

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    results.append([
        name,
        accuracy_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        matthews_corrcoef(y_test, y_pred)
    ])

# -----------------------------
# Results table
# -----------------------------
columns = [
    "Model",
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]

results_df = pd.DataFrame(results, columns=columns)
print(results_df)
