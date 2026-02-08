import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix

st.title("ML Assignment 2 – Classification Models")
st.write("Upload CSV test data and select a trained model")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if "diagnosis" not in data.columns:
        st.error("CSV must contain 'diagnosis' column")
        st.stop()

    X = data.drop(columns=["diagnosis"])
    y = data["diagnosis"].map({"M": 1, "B": 0})

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, f"{model_name}.pkl"))

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))
