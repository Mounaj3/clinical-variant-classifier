"""
interpretability.py
-------------------
Compute SHAP values and generate explanation plots
for the trained ClinVar classifier.

Usage:
    python interpretability.py
    python interpretability.py --model xgboost --n-samples 500
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROCESSED_DIR = "data/processed"
MODEL_DIR = "outputs/models"
SHAP_DIR = "outputs/shap"


def load_model_and_data(model_name):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    model = joblib.load(model_path)
    print(f"Model loaded: {model_path}")

    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    return model, X_test, y_test

def compute_shap_values(model, X, n_samples):
    if len(X) > n_samples:
        X = X.sample(n=n_samples, random_state=42).reset_index(drop=True)
        print(f"Using {n_samples} samples for SHAP computation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

   
    if len(shap_values.values.shape) == 3:
        print("Multi-class SHAP detected, keeping positive class (index 1)")
        shap_values = shap_values[..., 1]

    print(f"SHAP values computed — shape: {shap_values.values.shape}")
    return shap_values, X


def plot_summary_bar(shap_values, X_sample, output_dir):
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("Feature Importance (mean |SHAP|)")
    plt.tight_layout()
    path = os.path.join(output_dir, "shap_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_summary_beeswarm(shap_values, X_sample, output_dir):
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Summary — Feature Impact")
    plt.tight_layout()
    path = os.path.join(output_dir, "shap_beeswarm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_waterfall(shap_values, y_sample, output_dir, n_samples=3):
    for i in range(min(n_samples, len(shap_values))):
        shap.plots.waterfall(shap_values[i], show=False)
        label = "Pathogenic" if y_sample.iloc[i] == 1 else "Benign"
        plt.title(f"Sample {i} — True label: {label}")
        plt.tight_layout()
        path = os.path.join(output_dir, f"shap_waterfall_{i}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


def save_shap_csv(shap_values, X_sample, output_dir):
    shap_df = pd.DataFrame(
        shap_values.values,
        columns=[f"shap_{c}" for c in X_sample.columns]
    )
    path = os.path.join(output_dir, "shap_values.csv")
    shap_df.to_csv(path, index=False)
    print(f"Saved: {path}")


def main(model_name, n_samples):
    os.makedirs(SHAP_DIR, exist_ok=True)

    print("--- Loading model and data ---")
    model, X_test, y_test = load_model_and_data(model_name)

    print("\n--- Computing SHAP values ---")
    shap_values, X_sample = compute_shap_values(model, X_test, n_samples)
    y_sample = y_test.iloc[:len(X_sample)].reset_index(drop=True)

    print("\n--- Generating plots ---")
    plot_summary_bar(shap_values, X_sample, SHAP_DIR)
    plot_summary_beeswarm(shap_values, X_sample, SHAP_DIR)
    plot_waterfall(shap_values, y_sample, SHAP_DIR, n_samples=3)
    save_shap_csv(shap_values, X_sample, SHAP_DIR)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=["xgboost", "lightgbm"],
                        help="Model to explain (default: xgboost)")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of samples for SHAP computation (default: 500)")
    args = parser.parse_args()

    main(args.model, args.n_samples)