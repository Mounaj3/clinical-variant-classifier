# ClinVar Variant Pathogenicity Classifier

This project predicts whether a genetic variant is **pathogenic** or **benign**
using machine learning.

---

## About the data

We use [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/), a free public database from NCBI
that contains human genetic variants and their clinical significance.
The file is downloaded automatically, no account needed.

---

## How it works

1. We download the ClinVar dataset from the NCBI FTP server
2. We keep only variants labeled as Benign or Pathogenic
3. We extract features like variant length, chromosome, review score, and origin
4. We train a classifier (XGBoost or LightGBM) to predict pathogenicity
5. We use SHAP to explain which features influenced the predictions

---

## Setup
```bash
python -m venv var_class
source var_class/bin/activate
pip install -r requirements.txt
```

---

## Usage
```bash
# Step 1 — Download the data
python download_data.py

# Step 2 — Clean and prepare the data
python preprocessing.py --nrows 200000

# Step 3 — Train the model
python model.py --model lightgbm --n-folds 5
python model.py --model xgboost --n-folds 5

# Step 4 — Explain the model
python interpretability.py
```

---

## Results

After training, you will find in the `outputs/` folder:
- ROC and Precision-Recall curves
- Confusion matrix
- Feature importance plot
- SHAP plots explaining individual predictions

