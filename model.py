"""
model.py
--------
Train and evaluate a classifier on the preprocessed ClinVar data.

Usage:
    python model.py
    python model.py --model xgboost --n-estimators 300 --max-depth 6
    python model.py --model lightgbm
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "outputs"
MODEL_DIR = "outputs/models"


def load_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"Benign: {(y_train == 0).sum():,}  |  Pathogenic: {(y_train == 1).sum():,}")
    return X_train, X_test, y_train, y_test


def build_model(model_name, n_estimators, max_depth, scale_pos_weight):
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1
        )
    elif model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'xgboost' or 'lightgbm'.")


def cross_validate(model, X_train, y_train, n_folds):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(cv.split(X_train, y_train),
                                                      total=n_folds,
                                                      desc="Cross-validation")):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val   = X_train.iloc[val_idx]
        y_fold_val   = y_train.iloc[val_idx]

        model.fit(X_fold_train, y_fold_train)
        y_proba = model.predict_proba(X_fold_val)[:, 1]
        score = roc_auc_score(y_fold_val, y_proba)
        scores.append(score)

    scores = np.array(scores)
    print(f"\n{n_folds}-Fold CV AUC-ROC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"Folds: {[round(s, 4) for s in scores]}")
    return scores

def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr  = average_precision_score(y_test, y_proba)

    print(f"\n--- Test Set Results ---")
    print(f"AUC-ROC : {auc_roc:.4f}")
    print(f"AUC-PR  : {auc_pr:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Benign', 'Pathogenic'])}")

    return y_pred, y_proba, auc_roc, auc_pr


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Pathogenic"],
                yticklabels=["Benign", "Pathogenic"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_curves(y_test, y_proba, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0], name=model_name)
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[1], name=model_name)
    axes[0].set_title("ROC Curve")
    axes[1].set_title("Precision-Recall Curve")
    plt.suptitle(f"Model: {model_name}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"curves_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_feature_importance(model, feature_names, model_name, top_n=15):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Feature Importance - {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"feature_importance_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def save_model(model, model_name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    joblib.dump(model, path)
    print(f"Model saved: {path}")


def main(model_name, n_estimators, max_depth, n_folds):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("--- Loading data ---")
    X_train, X_test, y_train, y_test = load_data()

    # Handle class imbalance for XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance ratio: {scale_pos_weight:.2f}")

    print(f"\n--- Building {model_name} ---")
    model = build_model(model_name, n_estimators, max_depth, scale_pos_weight)

    print(f"\n--- Cross Validation ---")
    cross_validate(model, X_train, y_train, n_folds)

    print(f"\n--- Training ---")
    model.fit(X_train, y_train)

    print(f"\n--- Evaluation ---")
    y_pred, y_proba, auc_roc, auc_pr = evaluate(model, X_test, y_test)

    print(f"\n--- Saving outputs ---")
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_curves(y_test, y_proba, model_name)
    plot_feature_importance(model, X_train.columns.tolist(), model_name)
    save_model(model, model_name)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=["xgboost", "lightgbm"],
                        help="Model to train (default: xgboost)")
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="Number of trees (default: 300)")
    parser.add_argument("--max-depth", type=int, default=6,
                        help="Max tree depth (default: 6)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    args = parser.parse_args()

    main(args.model, args.n_estimators, args.max_depth, args.n_folds)