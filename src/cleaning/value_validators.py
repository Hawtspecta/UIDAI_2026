# src/cleaning/value_validators.py


def validate_non_negative(df, cols):
    for col in cols:
        if col in df.columns:
            df = df[df[col] >= 0]
    return df


def validate_age_consistency(df):
    """
    Enrollments: age_0_5 + age_5_17 + age_18_greater > 0
    """
    required = ["age_0_5", "age_5_17", "age_18_greater"]
    if not all(c in df.columns for c in required):
        return df

    total = df["age_0_5"] + df["age_5_17"] + df["age_18_greater"]
    return df[total > 0]


def validate_update_bounds(df, update_cols, enrol_cols):
    """
    Updates should never exceed enrollments by absurd margins.
    """
    if not all(c in df.columns for c in update_cols + enrol_cols):
        return df

    updates = df[update_cols].sum(axis=1)
    enrol = df[enrol_cols].sum(axis=1)

    # Allow 3x as extreme outlier tolerance
    return df[updates <= enrol * 3]


import pandas as pd
from pathlib import Path


def run_uidai_validations():
    """
    Runs basic UIDAI data validation checks:
    - Negative values
    - Failures greater than total enrollments
    - Missing districts or states
    """

    data_dir = Path("data/raw")
    files = list(data_dir.rglob("*.csv"))

    if not files:
        raise FileNotFoundError("No raw CSV files found for validation")

    total_rows = 0
    issues_found = 0

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        total_rows += len(df)

        # Check numeric columns
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

        for col in numeric_cols:
            if (df[col] < 0).any():
                print(f"⚠ Negative values detected in column {col} of file {f.name}")
                issues_found += 1

        # Check failures > total_enrollments
        if "failures" in df.columns and "total_enrollments" in df.columns:
            mask = df["failures"] > df["total_enrollments"]
            if mask.any():
                print(
                    f"⚠ Failures > total_enrollments in file {f.name}: {mask.sum()} rows"
                )
                issues_found += mask.sum()

        # Check missing districts/states
        for col in ["district", "state"]:
            if col in df.columns and df[col].isna().any():
                print(
                    f"⚠ Missing values in {col} column of file {f.name}: {df[col].isna().sum()} rows"
                )
                issues_found += df[col].isna().sum()

    print(f"✔ Total rows scanned: {total_rows}")
    if issues_found == 0:
        print("✔ No UIDAI validation issues found")
    else:
        print(f"⚠ Total validation issues found: {issues_found}")
