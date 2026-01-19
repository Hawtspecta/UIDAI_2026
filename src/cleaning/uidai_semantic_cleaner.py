# src/cleaning/uidai_semantic_cleaner.py

from .district_canonicalizer import canonicalize_districts
from .value_validators import (
    validate_non_negative,
    validate_age_consistency,
    validate_update_bounds,
)


def clean_enrollment(df):
    before = len(df)
    df = canonicalize_districts(df)
    df = validate_non_negative(df, ["age_0_5", "age_5_17", "age_18_greater"])
    df = validate_age_consistency(df)
    after = len(df)
    print(f"🧹 Enrollment cleaned: {before - after} rows removed")
    return df


def clean_demographic(df):
    before = len(df)
    df = canonicalize_districts(df)
    df = validate_non_negative(df, ["demo_age_5_17", "demo_age_17_"])
    after = len(df)
    print(f"🧹 Demographic cleaned: {before - after} rows removed")
    return df


def clean_biometric(df):
    before = len(df)
    df = canonicalize_districts(df)
    df = validate_non_negative(df, ["bio_age_5_17", "bio_age_17_"])
    after = len(df)
    print(f"🧹 Biometric cleaned: {before - after} rows removed")
    return df
