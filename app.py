import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.pkl")

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="ML Assignment 2 | Classification Models",
    page_icon="🧠",
    layout="wide"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
.main { background-color: #f5f7fa; }
.title { font-size: 38px; font-weight: bold; color: #2c3e50; text-align: center; }
.subtitle { font-size: 18px; color: #34495e; text-align: center; }
</style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown("<div class='title'>🧠 ML Assignment 2</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Breast Cancer Classification Models</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- Sidebar --------------------
st.sidebar.header("📂 Upload & Model Selection")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

model_name = st.sidebar.selectbox(
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

MODEL_FILES = {
    "Logistic Regression": "Logistic Regression.pkl",
    "Decision Tree": "Decision Tree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "Naive Bayes.pkl",
    "Random Forest": "Random Forest.pkl",
    "XGBoost": "XGBoost.pkl"
}

st.sidebar.info("""
**Dataset Requirements**
- Must contain `diagnosis`
- M → Malignant
- B → Benign
""")

# -------------------- Main Logic --------------------
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(data.head())

    if "diagnosis" not in data.columns:
        st.error("❌ 'diagnosis' column missing")
        st.stop()

    X = data.drop(columns=["diagnosis"])
    y = data["diagnosis"].map({"M": 1, "B": 0})

    scaler = load_scaler()

    if X.shape[1] != scaler.n_features_in_:
        st.error("❌ Feature mismatch with training data")
        st.stop()

    X_scaled = scaler.transform(X)

    model = load_model(f"model/{MODEL_FILES[model_name]}")
    y_pred = model.predict(X_scaled)

    # -------------------- Results --------------------
    st.markdown("## 📊 Model Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📑 Classification Report")
        report = classification_report(y, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    with col2:
        st.markdown("### 🔢 Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                    xticklabels=["Benign", "Malignant"],
                    yticklabels=["Benign", "Malignant"], ax=ax)
        st.pyplot(fig)

    st.success(f"✅ Prediction completed using **{model_name}**")

else:
    st.info("👈 Upload a CSV file to begin")
