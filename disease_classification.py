"""
Disease Classification Using Logistic Regression and SVM
Dataset: Breast Cancer Wisconsin (Diagnostic) -- breast_cancer_dataset.csv.xlsx
Target: diagnosis -> 0 = Healthy (Benign), 1 = Diseased (Malignant)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to files instead of requiring a display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)

# Paths are resolved relative to this file, so the script runs from any
# working directory and on any machine.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "breast_cancer_dataset.csv.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"


# =========================================================
# PART 1: DATASET EXPLORATION
# =========================================================
def explore_dataset(df):
    print("=" * 60)
    print("PART 1: DATASET EXPLORATION")
    print("=" * 60)
    print(f"Number of samples : {df.shape[0]}")
    print(f"Number of features: {df.shape[1] - 1}  (+1 target column: 'diagnosis')")
    print("\nFeature list:")
    for col in df.columns[:-1]:
        print(f"  - {col}")

    print("\nTarget variable: 'diagnosis'")
    print("  0 = Healthy  (Benign tumor)")
    print("  1 = Diseased (Malignant tumor)")
    print("\nClass distribution:")
    print(df["diagnosis"].value_counts().rename({0: "Healthy (0)", 1: "Diseased (1)"}))

    print("\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "  None found.")

    print("\nBasic statistics (first 5 features):")
    print(df.iloc[:, :5].describe())
    print()


# =========================================================
# PART 2: DATA PREPROCESSING
# =========================================================
def preprocess_data(df):
    print("=" * 60)
    print("PART 2: DATA PREPROCESSING")
    print("=" * 60)

    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.mean(numeric_only=True))
        print("Missing values were found and filled with column means.")
    else:
        print("No missing values found -> nothing to impute.")

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"Training samples: {X_train.shape[0]} (70%)")
    print(f"Testing samples : {X_test.shape[0]} (30%)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features standardized with StandardScaler (mean=0, std=1).\n")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


# =========================================================
# PART 3: MODEL DEVELOPMENT
# =========================================================
def train_models(X_train, y_train):
    print("=" * 60)
    print("PART 3: MODEL DEVELOPMENT")
    print("=" * 60)

    log_reg = LogisticRegression(max_iter=5000, random_state=42)
    log_reg.fit(X_train, y_train)
    print("Logistic Regression trained.")

    # CalibratedClassifierCV wraps SVC to provide predict_proba(); passing
    # probability=True directly to SVC is deprecated (sklearn >= 1.9).
    svm_clf = CalibratedClassifierCV(
        SVC(kernel="rbf", random_state=42), ensemble=False
    )
    svm_clf.fit(X_train, y_train)
    print("SVM (RBF kernel, calibrated for probabilities) trained.\n")

    return log_reg, svm_clf


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Healthy (0)", "Diseased (1)"]))
    cm = confusion_matrix(y_test, y_pred)
    return acc, y_pred, cm


# =========================================================
# PART 4: NEW PATIENT PREDICTION INTERFACE
# =========================================================
def predict_new_patient(features, scaler, log_reg, svm_clf, feature_names):
    """
    features: list/array of raw (unscaled) feature values in the same
              order as feature_names (30 values for this dataset).
    """
    features = np.array(features).reshape(1, -1)
    if features.shape[1] != len(feature_names):
        raise ValueError(
            f"Expected {len(feature_names)} features, got {features.shape[1]}."
        )
    features_df = pd.DataFrame(features, columns=feature_names)
    features_scaled = scaler.transform(features_df)

    lr_pred = log_reg.predict(features_scaled)[0]
    svm_pred = svm_clf.predict(features_scaled)[0]
    lr_prob = log_reg.predict_proba(features_scaled)[0][1]
    svm_prob = svm_clf.predict_proba(features_scaled)[0][1]

    label = {0: "Healthy (0)", 1: "Diseased (1)"}
    print("New patient prediction:")
    print(f"  Logistic Regression -> {label[lr_pred]}  (P(diseased)={lr_prob:.3f})")
    print(f"  SVM                 -> {label[svm_pred]}  (P(diseased)={svm_prob:.3f})")
    return lr_pred, svm_pred


def interactive_new_patient(scaler, log_reg, svm_clf, feature_names):
    """Prompts the user to type in a new patient's measurements one by one."""
    print("\nEnter the new patient's measurements below.")
    print("(Press Ctrl+C or Ctrl+Z then Enter at any point to cancel.)\n")
    values = []
    try:
        for name in feature_names:
            val = float(input(f"  {name}: "))
            values.append(val)
    except (EOFError, KeyboardInterrupt, ValueError):
        print("\nInteractive input cancelled or unavailable — skipping.")
        return None
    return predict_new_patient(values, scaler, log_reg, svm_clf, feature_names)


# =========================================================
# PART 5: VISUALIZATION
# =========================================================
def make_visualizations(df, y_test, lr_pred, svm_pred, cm_lr, cm_svm,
                         log_reg, svm_clf, X_test, acc_lr, acc_svm):
    print("=" * 60)
    print("PART 5: VISUALIZATION")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Class distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(x=df["diagnosis"].map({0: "Healthy", 1: "Diseased"}))
    plt.title("Class Distribution")
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_class_distribution.png", dpi=150)
    plt.close()

    # 2. Correlation heatmap (subset: mean-value features, for readability)
    mean_cols = [c for c in df.columns if c.startswith("mean")] + ["diagnosis"]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[mean_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap (mean-value features)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_correlation_heatmap.png", dpi=150)
    plt.close()

    # 3. Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cm, title in zip(axes, [cm_lr, cm_svm],
                              ["Logistic Regression", "SVM"]):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Healthy", "Diseased"],
                    yticklabels=["Healthy", "Diseased"], ax=ax)
        ax.set_title(f"{title} - Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_confusion_matrices.png", dpi=150)
    plt.close()

    # 4. ROC curves
    plt.figure(figsize=(6, 5))
    for model, name in [(log_reg, "Logistic Regression"), (svm_clf, "SVM")]:
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_roc_curves.png", dpi=150)
    plt.close()

    # 5. Accuracy comparison bar chart
    plt.figure(figsize=(5, 4))
    plt.bar(["Logistic Regression", "SVM"], [acc_lr, acc_svm],
            color=["#4C72B0", "#DD8452"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    for i, v in enumerate([acc_lr, acc_svm]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_accuracy_comparison.png", dpi=150)
    plt.close()

    # 6. Decision boundary using PCA (2D projection, illustrative only)
    X_all = df.drop(columns=["diagnosis"])
    y_all = df["diagnosis"]
    scaler_full = StandardScaler().fit(X_all)
    X_all_scaled = scaler_full.transform(X_all)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all_scaled)

    lr_2d = LogisticRegression(max_iter=5000).fit(X_pca, y_all)
    svm_2d = SVC(kernel="rbf").fit(X_pca, y_all)

    xx, yy = np.meshgrid(
        np.linspace(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 300),
        np.linspace(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 300),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model, title in zip(axes, [lr_2d, svm_2d],
                                 ["Logistic Regression", "SVM"]):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_all, cmap="coolwarm",
                   edgecolor="k", s=15)
        ax.set_title(f"{title} Decision Boundary (PCA 2D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/6_decision_boundaries_pca.png", dpi=150)
    plt.close()

    print(f"All plots saved to the '{OUTPUT_DIR}/' folder:")
    print("  1_class_distribution.png")
    print("  2_correlation_heatmap.png")
    print("  3_confusion_matrices.png")
    print("  4_roc_curves.png")
    print("  5_accuracy_comparison.png")
    print("  6_decision_boundaries_pca.png\n")


# =========================================================
# PART 6: COMPARISON AND DISCUSSION
# =========================================================
def compare_models(acc_lr, acc_svm):
    print("=" * 60)
    print("PART 6: COMPARISON AND DISCUSSION")
    print("=" * 60)
    print(f"Logistic Regression accuracy: {acc_lr:.4f}")
    print(f"SVM accuracy                : {acc_svm:.4f}")

    if acc_lr > acc_svm:
        winner = "Logistic Regression"
    elif acc_svm > acc_lr:
        winner = "SVM"
    else:
        winner = "Both models (tied)"
    print(f"\nHigher accuracy: {winner}")

    print("""
Discussion:
- Which model achieved higher accuracy?
  See result above (values differ slightly run-to-run only if random_state
  changes; with random_state=42 the result is deterministic).

- Why might one model perform better?
  The breast cancer dataset's classes are close to linearly separable in
  the standardized feature space (features like 'mean radius', 'mean
  concave points' correlate strongly and monotonically with malignancy),
  which favors Logistic Regression's linear decision boundary and gives it
  interpretable coefficients. SVM with an RBF kernel can capture non-linear
  boundaries and is often slightly more robust to feature overlap /
  outliers, which can give it an edge, especially after standardization.
  In practice the two models perform very similarly on this dataset
  (>=95% accuracy for both), so the difference usually comes down to
  precision/recall trade-offs on the minority (diseased) class rather
  than raw accuracy.

- Recommendation for this medical problem:
  For a diagnostic screening tool, Logistic Regression is often preferred
  in medical settings because:
    1) It outputs directly interpretable probabilities and coefficients,
       letting clinicians see which features drive a prediction.
    2) It is simpler to validate, audit, and explain to regulators and
       patients (important for medical/legal accountability).
    3) It is computationally cheaper and scales well.
  SVM can be preferred if the goal is squeezing out marginally higher
  accuracy/recall and the model does not need to be interpreted by a
  clinician (e.g., an internal automated triage step). In this assignment,
  given both models perform comparably, Logistic Regression is the
  recommended choice for a clinical decision-support context because of
  its interpretability, with SVM as a strong secondary check.
""")


# =========================================================
# MAIN
# =========================================================
def main():
    df = pd.read_excel(DATA_FILE)

    explore_dataset(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    log_reg, svm_clf = train_models(X_train, y_train)

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    acc_lr, lr_pred, cm_lr = evaluate_model("Logistic Regression", log_reg, X_test, y_test)
    acc_svm, svm_pred, cm_svm = evaluate_model("SVM", svm_clf, X_test, y_test)

    print("=" * 60)
    print("PART 4: NEW PATIENT PREDICTION")
    print("=" * 60)
    # Example 1: raw measurements resembling a benign case
    example_healthy = [
        12.0, 15.0, 78.0, 450.0, 0.09, 0.08, 0.03, 0.02, 0.17, 0.06,
        0.3, 1.0, 2.0, 25.0, 0.006, 0.02, 0.02, 0.01, 0.02, 0.003,
        13.5, 20.0, 87.0, 550.0, 0.13, 0.18, 0.12, 0.07, 0.28, 0.08,
    ]
    # Example 2: raw measurements resembling a malignant case
    example_diseased = [
        20.5, 21.0, 135.0, 1300.0, 0.11, 0.22, 0.25, 0.15, 0.20, 0.06,
        0.8, 1.2, 5.5, 120.0, 0.007, 0.03, 0.04, 0.02, 0.02, 0.004,
        25.0, 28.0, 165.0, 1900.0, 0.15, 0.45, 0.5, 0.22, 0.32, 0.09,
    ]
    print("\nExample patient A (expected: healthy-like measurements):")
    predict_new_patient(example_healthy, scaler, log_reg, svm_clf, feature_names)
    print("\nExample patient B (expected: diseased-like measurements):")
    predict_new_patient(example_diseased, scaler, log_reg, svm_clf, feature_names)

    # Uncomment to type in a real patient's values from the keyboard:
    # interactive_new_patient(scaler, log_reg, svm_clf, feature_names)

    make_visualizations(df, y_test, lr_pred, svm_pred, cm_lr, cm_svm,
                         log_reg, svm_clf, X_test, acc_lr, acc_svm)

    compare_models(acc_lr, acc_svm)


if __name__ == "__main__":
    main()
