# src/cleaning/district_canonicalizer.py

DISTRICT_MAP = {
    # Karnataka
    "Bangalore Urban": "Bengaluru Urban",
    "Bangalore Rural": "Bengaluru Rural",
    "Bijapur(Kar)": "Vijayapura",

    # Maharashtra 
    "Mumbai Suburban": "Mumbai",
    "Mumbai( Sub Urban )": "Mumbai",
    "Ahmed Nagar": "Ahmednagar",
    "Ahmadnagar": "Ahmednagar",
    "Chatrapati Sambhaji Nagar": "Chhatrapati Sambhajinagar",
    "Chhatrapati Sambhaji Nagar": "Chhatrapati Sambhajinagar",

    # UP cleanup
    "Gautam Buddha Nagar *": "Gautam Buddha Nagar",

    # Haryana
    "Gurgaon": "Gurugram",

    # Bihar
    "Aurangabad(Bh)": "Aurangabad",

    # West Bengal
    "North 24 Parganas": "24 North Parganas",
    "South 24 Parganas": "24 South Parganas",

    # Odisha
    "Orissa": "Odisha",
    "Anugul": "Angul",
    "Angul": "Angul",
    "Balangir": "Balangir",

    # Telangana / AP split
    "Ranga Reddy": "Rangareddy",
    "Warangal (Urban)": "Warangal",
    "Warangal Rural": "Warangal",
    "Warangal Urban": "Warangal",

    # Dadra & Nagar Haveli variants
    "Dadra & Nagar Haveli": "Dadra And Nagar Haveli",
    "Dadra and Nagar Haveli": "Dadra And Nagar Haveli",
    

    # Delhi
    "Central Delhi": "Delhi",
    "East Delhi": "Delhi",
    "North Delhi": "Delhi",
    "South Delhi": "Delhi",
    "West Delhi": "Delhi",

    # Generic noise
    "Urban": None,
    "Rural": None
}


ADDRESS_PATTERNS = [
    r"\bNear\b",
    r"\bCross\b",
    r"\bRoad\b",
    r"\bRd\b",
    r"\bStreet\b",
    r"\bSt\b",
    r"\bHospital\b",
    r"\bThana\b",
    r"\bGarden\b",
    r"\bLayout\b",
    r"\bPhase\b",
    r"\bSector\b",
    r"\bBlock\b",
    r"\bColony\b"
]

def canonicalize_districts(df):
    df['district'] = df['district'].astype(str).str.strip().str.title()

    # Identify address-like entries
    address_regex = '|'.join(ADDRESS_PATTERNS)
    mask_address = df['district'].str.contains(
        address_regex,
        regex=True,
        na=False
    )

    # Force invalid districts to NULL
    df.loc[mask_address, 'district'] = None

    # Apply canonical district mapping
    df['district'] = df['district'].replace(DISTRICT_MAP)

    return df


import pandas as pd
from pathlib import Path

def is_address_like(series):
    return series.str.contains(
        '|'.join(ADDRESS_PATTERNS),
        regex=True,
        na=False
    )


def audit_districts():
    """
    Audits district names before and after canonicalization
    to ensure UIDAI semantic consistency.
    """

    data_dir = Path("data/raw")
    files = list(data_dir.rglob("*.csv"))

    if not files:
        raise FileNotFoundError("No raw CSV files found for district audit")

    raw_districts = set()

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        if 'district' in df.columns:
            raw_districts.update(df['district'].dropna().unique())

    raw_districts = sorted(raw_districts)

    df_audit = pd.DataFrame({'district': raw_districts})
    df_audit = df_audit[
        ~df_audit['district'].str.match(r"^\d+.*", na=False)
    ]

    df_audit['canonical'] = (
        df_audit['district']
        .astype(str)
        .str.strip()
        .str.title()
        .replace(DISTRICT_MAP)
    )

    # Drop rows where canonical district is null or empty
    df_audit = df_audit[
        df_audit['canonical'].notna() &
        (df_audit['canonical'].str.lower() != 'none')
    ]

    df_audit = df_audit[~is_address_like(df_audit['district'])]

    merged = df_audit[df_audit['district'] != df_audit['canonical']]

    print(f"✔ Total raw districts found: {len(raw_districts)}")
    print(f"✔ Canonical merges applied: {len(merged)}")

    if not merged.empty:
        print("✔ Sample merges:")
        print(merged.head(10).to_string(index=False))

    VALID_SUFFIXES = [
        "Bengaluru Urban",
        "Bengaluru Rural"
    ]

    unresolved = df_audit[
        df_audit['canonical'].str.contains(
            r"\bNear\b|\bRoad\b|\bCross\b|\bHospital\b|\bThana\b",
            regex=True,
            na=False
        ) 
    ]


    if unresolved.empty:
        print("✔ No unresolved UIDAI-style district patterns found")
    else:
        print("⚠ Unresolved district patterns detected:")
        print(unresolved.head(10).to_string(index=False))

