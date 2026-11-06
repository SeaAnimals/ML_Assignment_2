import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🧬", layout="wide")
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
DEFAULT_FILE = ROOT / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
TARGET = "diagnosis"
POSITIVE_LABEL = "malignant"

st.title("Breast Cancer Classification Dashboard")
st.caption("Upload a labeled test CSV, select a saved classifier, and inspect its evaluation.")
st.info("This app evaluates saved models only. It does not retrain models on uploaded test data.")

use_default = st.checkbox("Use default test data file")
uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"], disabled=use_default)

if use_default and DEFAULT_FILE.exists():
    data = pd.read_csv(DEFAULT_FILE)
elif uploaded is not None:
    st.success(f"Using uploaded file: `{uploaded.name}`")
    try:
        data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
else:
    st.write("Start by uploading test data or use the default test data (test_data.csv) from the project repository.")
    st.stop()

if TARGET not in data.columns:
    st.error(f"CSV must include the target column `{TARGET}`.")
    st.stop()

labels = data[TARGET].astype(str).str.lower().str.strip()
if not set(labels.unique()).issubset({"benign", "malignant"}):
    st.error("The diagnosis column must contain only 'benign' and 'malignant'.")
    st.stop()

X = data.drop(columns=[TARGET])
y = (labels == POSITIVE_LABEL).astype(int)

model_name = st.selectbox("Select classifier", list(MODEL_FILES.keys()))
model_path = MODEL_DIR / MODEL_FILES[model_name]
if not model_path.exists():
    st.error(f"Saved model file is missing: {model_path.name}")
    st.stop()

try:
    model = joblib.load(model_path)
    expected = list(getattr(model, "feature_names_in_", []))
    if expected and list(X.columns) != expected:
        missing = sorted(set(expected) - set(X.columns))
        extra = sorted(set(X.columns) - set(expected))
        st.error(f"Feature columns/order do not match the training data. Missing: {missing}; extra: {extra}.")
        st.stop()
    X = X.apply(pd.to_numeric, errors="raise")
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
except Exception as exc:
    st.error(f"Model evaluation failed. Check feature names, columns and numeric values. Details: {exc}")
    st.stop()

metrics = {
    "Accuracy": accuracy_score(y, pred),
    "AUC": roc_auc_score(y, prob) if y.nunique() == 2 else float("nan"),
    "Precision": precision_score(y, pred, zero_division=0),
    "Recall": recall_score(y, pred, zero_division=0),
    "F1 Score": f1_score(y, pred, zero_division=0),
    "MCC": matthews_corrcoef(y, pred),
}
st.subheader(f"Evaluation: {model_name}")
cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.4f}" if pd.notna(value) else "N/A")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(
        y, pred, labels=[0, 1], target_names=["Benign", "Malignant"],
        zero_division=0, output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

st.subheader("Prediction preview")
preview = data.copy()
preview["predicted_diagnosis"] = ["malignant" if p == 1 else "benign" for p in pred]
preview["malignant_probability"] = prob
st.dataframe(preview.head(20), use_container_width=True)
st.download_button(
    "Download predictions CSV",
    preview.to_csv(index=False).encode("utf-8"),
    file_name="classification_predictions.csv",
    mime="text/csv"
)
