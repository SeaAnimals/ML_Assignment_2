# Breast Cancer Classification — ML Assignment 2
- GitHub repository: https://github.com/SeaAnimals/ML_Assignment_2
- Live app: https://steel-plate-fault-classifier-gz6crdafengjsxp8qyzmx4.streamlit.app/

## a. Problem statement
Build and compare five supervised classification algorithms to predict whether a breast-mass sample is **malignant** or **benign** from numeric measurements computed from a digitized fine-needle aspirate (FNA) image. The positive class is defined as `malignant`.

## b. Dataset description
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC).
- **Source:** UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic) dataset.
- **Instances:** 569.
- **Input features:** 30 numeric measurements.
- **Target:** `diagnosis` (`benign` / `malignant`).
- **Problem type:** Binary classification.
- **Split:** Stratified 80% training / 20% test, `random_state=42`.
- **Preprocessing:** StandardScaler is fitted only within the training pipeline for Logistic Regression, KNN and SVM. Decision Tree and Gaussian Naive Bayes use the original numeric feature values.
- **Note:** The CSV in `test_data.csv` is the held-out test partition and includes the target column, as required by the assignment app.

## c. GitHub Repository Link

https://github.com/SeaAnimals/ML_Assignment_2

## d. Models used

All five models are trained on the same training partition and evaluated on the same held-out test partition. AUC uses the predicted probability for the malignant class. Precision, recall and F1 use malignant as the positive class.

### Comparison table

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9649 | 0.9960 |      0.9750 |   0.9286 | 0.9512 | 0.9245 |
| Decision Tree          |     0.9123 | 0.8770 |      0.9444 |   0.8095 | 0.8718 | 0.8113 |
| K-Nearest Neighbors    |     0.9561 | 0.9823 |      0.9744 |   0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes   |     0.9386 | 0.9934 |      1.0000 |   0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine |     0.9737 | 0.9947 |      1.0000 |   0.9286 | 0.9630 | 0.9442 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | It identifies 39 of the 42 malignant cases (recall 0.93) and correctly labels 39 of the 40 samples it predicts as malignant (precision 0.98). It misses 3 malignant cases and produces 1 false alarm. Its excellent AUC (0.996) indicates that it separates the two classes very well across decision thresholds. |
| Decision Tree | It correctly identifies 34 of the 42 malignant cases (recall 0.81), but misses 8 malignant cases. It produces 2 false alarms. Its lower AUC (0.877) and MCC (0.811) show that this tree configuration separates benign and malignant samples less consistently than the other models. |
| K-Nearest Neighbors | It detects 38 of the 42 malignant cases (recall 0.90), missing 4, and produces 1 false alarm. The strong accuracy (0.956) and AUC (0.982) suggest that nearby samples usually have similar diagnoses, although it is slightly less effective than SVM and Logistic Regression on this split. |
| Gaussian Naive Bayes | It has perfect precision (1.00): none of the samples predicted as malignant are actually benign. However, it detects only 35 of the 42 malignant cases (recall 0.83), missing 7. This conservative positive prediction behavior explains its lower F1 score (0.909), despite a strong AUC (0.993). |
| Support Vector Machine | It gives the highest accuracy (0.974), F1 (0.963) and MCC (0.944). It identifies 39 of the 42 malignant cases (recall 0.93) with perfect precision (1.00), so it makes no false-positive malignant predictions and misses 3 malignant cases. It is the strongest overall classifier by these threshold-based metrics on this test split. |


**Overall winner on this test split:** **Logistic Regression**, selected by highest AUC (with F1 and MCC used as tie-breakers). This winner is specific to this dataset and split; inspect recall and false negatives as well

## Project structure

```text
ML_Assignment_2_Project/
├── README.md
├── app.py
├── requirements.txt
├── test_data.csv
├── trainer.py
├── data/
│   └── breast_cancer_dataset.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knearest_neighbors.joblib
│   ├── gaussian_naive_bayes.joblib
│   └── support_vector_machine.joblib
└── results/
    └── model_comparison.csv
```

## Run locally

```
pip install -r requirements.txt
python trainer.py
streamlit run app.py
```

## Streamlit Community Cloud deployment

- upload a CSV (or tick `Use default test data file` to use test_data.csv in the repository) and pick a model from the dropdown.
