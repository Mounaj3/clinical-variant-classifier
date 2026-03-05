"""
preprocessing.py
----------------
Load and clean ClinVar data, engineer features,
and split into train/test sets.

Usage:
    python preprocessing.py
    python preprocessing.py --assembly GRCh38 --test-size 0.2 --nrows 20000
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


RAW_PATH = "data/raw/variant_summary.txt"
PROCESSED_DIR = "data/processed"

BENIGN_LABELS = ["Benign", "Likely benign"]
PATHOGENIC_LABELS = ["Pathogenic", "Likely pathogenic"]


def load_data(assembly, nrows=None):
    print(f"Loading data from {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH, sep="\t", low_memory=False, on_bad_lines="skip", nrows=nrows)
    print(f"  {df.shape[0]:,} rows loaded")

    df = df[df["Assembly"] == assembly]
    print(f"  {df.shape[0]:,} rows after filtering assembly={assembly}")
    return df


def filter_classes(df):
    all_labels = BENIGN_LABELS + PATHOGENIC_LABELS
    df = df[df["ClinicalSignificance"].isin(all_labels)].copy()

    df["label"] = df["ClinicalSignificance"].apply(
        lambda x: 1 if x in PATHOGENIC_LABELS else 0
    )

    print(f"  {df.shape[0]:,} rows after class filtering")
    print(f"  Benign: {(df['label'] == 0).sum():,}  |  Pathogenic: {(df['label'] == 1).sum():,}")
    return df


def filter_chromosomes(df):
    valid_chroms = [str(i) for i in range(1, 23)] + ["X", "Y"]
    df = df[df["Chromosome"].isin(valid_chroms)].copy()
    print(f"  {df.shape[0]:,} rows after chromosome filtering")
    return df


def engineer_features(df):
    # Variant length
    df["variant_length"] = (df["Stop"] - df["Start"]).abs().fillna(0).astype(int)

    # Chromosome as integer
    chrom_map = {str(i): i for i in range(1, 23)}
    chrom_map.update({"X": 23, "Y": 24})
    df["chrom_int"] = df["Chromosome"].map(chrom_map).fillna(0).astype(int)

    # Is sex chromosome
    df["is_sex_chrom"] = df["Chromosome"].isin(["X", "Y"]).astype(int)

    # Number of associated phenotypes
    df["n_phenotypes"] = df["PhenotypeList"].apply(
        lambda x: len(str(x).split(";")) if pd.notna(x) else 0
    )

    # Review status as a score (ClinVar star rating)
    review_map = {
        "practice guideline": 4,
        "reviewed by expert panel": 3,
        "criteria provided, multiple submitters, no conflicts": 2,
        "criteria provided, single submitter": 1,
        "no assertion criteria provided": 0,
        "no assertion provided": 0,
    }
    df["review_score"] = df["ReviewStatus"].str.lower().map(review_map).fillna(0).astype(int)

    # Variant type encoded
    df["type_enc"] = df["Type"].astype("category").cat.codes

    # Origin encoded
    origin_map = {
        "germline": 0, "somatic": 1, "de novo": 2,
        "inherited": 3, "unknown": 4, "not provided": 4
    }
    df["origin_enc"] = df["Origin"].str.lower().map(origin_map).fillna(4).astype(int)

    return df


def select_features(df):
    features = [
        "Start", "Stop", "variant_length",
        "chrom_int", "is_sex_chrom",
        "n_phenotypes", "review_score",
        "type_enc", "origin_enc",
    ]
    features = [f for f in features if f in df.columns]
    return df[features], features


def split_and_save(X, y, test_size, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    print(f"\nTrain: {X_train.shape[0]:,} samples")
    print(f"Test:  {X_test.shape[0]:,} samples")
    print(f"Saved to {PROCESSED_DIR}/")


def main(assembly, test_size, nrows):
    print("--- Preprocessing ---")

    df = load_data(assembly, nrows=nrows)
    df = filter_classes(df)
    df = filter_chromosomes(df)
    df = engineer_features(df)

    X, features = select_features(df)
    y = df["label"]

    print(f"\nFeatures: {features}")
    split_and_save(X, y, test_size)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assembly", type=str, default="GRCh38",
                        help="Genome assembly to use (default: GRCh38)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set proportion (default: 0.2)")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Number of rows to load (default: 20000)")
    args = parser.parse_args()

    main(args.assembly, args.test_size, args.nrows)