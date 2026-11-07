# Question/Step 1: Import libraries required for data handling, modeling and evaluation.
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

RANDOM_STATE = 42
PROJECT_DIR = Path(".")
MODEL_DIR = PROJECT_DIR / "model"
RESULTS_DIR = PROJECT_DIR / "results"
DATASET_FILE = PROJECT_DIR / "data" / "breast_cancer_dataset.csv"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Question/Step 1: Load the public Wisconsin Diagnostic Breast Cancer dataset.
# It contains 569 samples and 30 numeric features.
# 1. Read the dataset from your local path
df = pd.read_csv(DATASET_FILE)
X = df.drop(columns=["diagnosis"]).copy()
# Recode text labels back to numbers: malignant as positive class 1, benign as 0
y = df["diagnosis"].map({"benign": 0, "malignant": 1}).astype(int)
# Split into train and test sets (remains exactly the same)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print("Dataset shape:", df.shape)
print("Number of input features:", X.shape[1])
print("Class counts:")
print(df["diagnosis"].value_counts())

# Validate the assignment's minimum dataset size requirements.
assert X.shape[1] >= 12, "Dataset must have at least 12 features."
assert X.shape[0] >= 500, "Dataset must have at least 500 instances."
assert df["diagnosis"].nunique() == 2, "This script expects a binary classification target."
print("Dataset meets the minimum feature and instance requirements.")

# Split data into training and held-out test partitions.
# Stratification preserves the class ratio in both partitions.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("\n\nTraining rows:", len(X_train))
print("Test rows:", len(X_test))

# Define all five required classifiers.
# Scaling is included in pipelines for LR, KNN and SVM to avoid fitting preprocessing on test data.
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        # Wrap SVC inside CalibratedClassifierCV to get probabilities without the warning
        ("classifier", CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", random_state=RANDOM_STATE), 
            ensemble=False
        ))
    ])
}
print("\n\nModels:", list(models.keys()))


# Train each classifier and calculate all six required evaluation metrics.
# AUC uses predicted probability of malignant (positive class=1).
results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    results.append({
        "ML Model Name": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_probability),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

metrics_df = pd.DataFrame(results).sort_values("AUC", ascending=False)
print(f"\nModel Performance Metrics:\n{metrics_df.round(4).to_string()}")

# Select the best model on this held-out test set.
# Primary selection criterion is AUC; F1 and MCC are tie-breakers.
winner_row = metrics_df.sort_values(["AUC", "F1", "MCC"], ascending=False).iloc[0]
print("\n\nOverall winner on this test split:", winner_row["ML Model Name"])
print("This winner is specific to this dataset and split; inspect recall and false negatives as well.")

# Save each trained model for reuse in the Streamlit app.
# The full preprocessing pipeline is saved together with its classifier.
for model_name, model in models.items():
    filename = model_name.lower().replace(" ", "_").replace("-", "")
    joblib.dump(model, MODEL_DIR / f"{filename}.joblib")
print("Saved model files:", sorted(p.name for p in MODEL_DIR.glob("*.joblib")))

# Question/Step 3: Save the held-out test data with its target column for app evaluation.
test_data = X_test.copy()
test_data["diagnosis"] = pd.Series(y_test, index=y_test.index).map({0: "benign", 1: "malignant"})
test_data.to_csv(PROJECT_DIR / "test_data.csv", index=False)
metrics_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
