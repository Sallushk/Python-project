"""Read, audit and prepare the supplied CSV without changing the raw file."""

import os
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "stroke"
]
NUMERIC_COLUMNS = ["id", "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"]
CONTINUOUS_COLUMNS = ["age", "avg_glucose_level", "bmi"]
KNOWN_SMOKING = ["formerly smoked", "never smoked", "smokes"]
SMOKING_ORDER = KNOWN_SMOKING + ["Unknown"]
FACTOR_NAMES = {
    "age": "Age", "avg_glucose_level": "Average glucose", "bmi": "BMI",
    "hypertension": "Hypertension", "smoking_status": "Smoking status"
}
ALLOWED_CATEGORIES = {
    "gender": ["Female", "Male", "Other"],
    "ever_married": ["Yes", "No"],
    "work_type": ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    "Residence_type": ["Urban", "Rural"],
    "smoking_status": SMOKING_ORDER
}
# Physiologically plausible hard limits used in better-cleaning step
PLAUSIBLE_LIMITS = {
    "age":               (0,   120),
    "avg_glucose_level": (30,  600),
    "bmi":               (10,  80),
}


def read_dataset(file_path):
    """N/A BMI cells are missing measurements, not zeroes."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    data = pd.read_csv(file_path, na_values=["N/A", "N/a", "n/a", "NA", "na", "", " "])
    if data.empty:
        raise ValueError("The dataset contains no records.")
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Required columns missing: {', '.join(missing)}")
    return data


def display_dataset_info(data):
    print(f"\nRAW DATASET INFO  Rows: {data.shape[0]}  Columns: {data.shape[1]}")
    print("Columns:", ", ".join(data.columns))
    print("\nFirst five records:\n", data.head().to_string(index=False))
    print("\nData types:\n", data.dtypes.to_string())
    print("\nMissing values:\n", data.isna().sum().to_string())
    print("\nExact duplicates:", data.duplicated().sum(),
          "  Repeated IDs:", data["id"].duplicated().sum())
    print("\nStroke counts (0=no, 1=yes):")
    print(data["stroke"].value_counts().sort_index().to_string())


def _strip_strings(data):
    """Strip leading/trailing whitespace from all object columns."""
    str_cols = data.select_dtypes(include="object").columns
    for col in str_cols:
        data[col] = data[col].str.strip()
    return data


def _flag_implausible(data):
    """
    Add boolean flag columns for values outside physiological bounds.
    Flags are informational only — records are RETAINED.
    """
    for col, (lo, hi) in PLAUSIBLE_LIMITS.items():
        flag_col = col + "_implausible"
        data[flag_col] = data[col].notna() & ((data[col] < lo) | (data[col] > hi))
        flagged = data[flag_col].sum()
        if flagged:
            print(f"  [cleaning] {flagged} implausible value(s) flagged in '{col}' "
                  f"(outside [{lo}, {hi}]) — retained with flag column '{flag_col}'.")
    return data


def validate_and_clean(raw, impute_bmi=False):
    """
    Reject invalid input rather than silently dropping or relabeling records.

    Parameters
    ----------
    impute_bmi : bool
        If True, fill missing BMI with the group median (by stroke status).
        Default False keeps original behaviour (missing BMI is allowed).
    """
    data = raw.copy()

    # 1. Strip whitespace from string columns
    data = _strip_strings(data)

    # 2. Coerce numeric columns
    for col in NUMERIC_COLUMNS:
        conv = pd.to_numeric(data[col], errors="coerce")
        if (data[col].notna() & conv.isna()).any():
            raise ValueError(f"Non-numeric values in {col}")
        if not np.isfinite(conv.dropna().to_numpy()).all():
            raise ValueError(f"Infinite values in {col}")
        data[col] = conv

    # 3. Required columns must not be null (except bmi which may be missing)
    for col in REQUIRED_COLUMNS:
        if col != "bmi" and data[col].isna().any():
            raise ValueError(f"Missing values in {col}; inspect the CSV.")

    # 4. Binary flag columns
    for col in ["hypertension", "heart_disease", "stroke"]:
        if not data[col].isin([0, 1]).all():
            raise ValueError(f"{col} must be 0 or 1.")

    # 5. Range / sign checks
    if (data["age"] < 0).any():
        raise ValueError("Age cannot be negative.")
    if (data["avg_glucose_level"] <= 0).any() or (data["bmi"].dropna() <= 0).any():
        raise ValueError("Glucose and BMI must be positive.")
    if (data["id"] <= 0).any() or (data["id"] % 1 != 0).any():
        raise ValueError("IDs must be positive whole numbers.")

    # 6. Allowed category check
    for col, cats in ALLOWED_CATEGORIES.items():
        bad = data.loc[~data[col].isin(cats), col].unique().tolist()
        if bad:
            raise ValueError(f"Unexpected categories in {col}: {bad}")

    # 7. Flag physiologically implausible values (retained)
    data = _flag_implausible(data)

    # 8. Optional BMI imputation (group median by stroke status)
    if impute_bmi and data["bmi"].isna().any():
        medians = data.groupby("stroke")["bmi"].transform("median")
        overall_median = data["bmi"].median()
        data["bmi"] = data["bmi"].fillna(medians).fillna(overall_median)
        data["bmi_imputed"] = data["bmi"].isna()   # will be all-False after fill
        print(f"  [cleaning] BMI imputed with group median where missing.")

    # 9. Remove exact duplicates; raise on conflicting IDs
    dupes = int(data.duplicated().sum())
    data = data.drop_duplicates()
    if data["id"].duplicated().any():
        raise ValueError("Repeated IDs with conflicting records; inspect before analysis.")
    return data, dupes


def make_populations(data):
    # Drop flag columns before splitting so downstream code stays clean
    flag_cols = [c for c in data.columns if c.endswith("_implausible") or c == "bmi_imputed"]
    clean = data.drop(columns=flag_cols, errors="ignore")
    adults = clean.loc[clean["age"] >= 18].copy()
    complete = adults.dropna(subset=["bmi"])
    complete = complete.loc[complete["smoking_status"].isin(KNOWN_SMOKING)].copy()
    return {"All ages": clean.copy(), "Adults": adults, "Complete adults": complete}


def add_age_bands(adults):
    data = adults.copy()
    bins = [18, 30, 40, 50, 60, 70, 80, np.inf]
    labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    data["age_band"] = pd.cut(data["age"], bins=bins, labels=labels, right=False)
    return data


def audit_tables(raw, populations, duplicates_removed):
    adults = populations["Adults"]
    audit_rows = [
        ("Raw records", len(raw)), ("Raw columns", len(raw.columns)),
        ("Exact duplicates removed", duplicates_removed),
        ("Raw repeated IDs", int(raw["id"].duplicated().sum())),
        ("All-age records after cleaning", len(populations["All ages"])),
        ("Records under 18", int((populations["All ages"]["age"] < 18).sum())),
        ("Adult records", len(adults)),
        ("Adult stroke records", int((adults["stroke"] == 1).sum())),
        ("Adult BMI missing", int(adults["bmi"].isna().sum())),
        ("Adult smoking Unknown", int((adults["smoking_status"] == "Unknown").sum())),
        ("Complete adult records", len(populations["Complete adults"]))
    ]
    missing_rows = [
        {"population": pop, "stroke": s, "issue": issue, "issue_count": cnt,
         "group_total": len(g), "percent": 100 * cnt / len(g) if len(g) > 0 else np.nan}
        for pop, df in populations.items()
        for s in [0, 1]
        for g in [df.loc[df["stroke"] == s]]
        for issue, cnt in [
            ("BMI missing", int(g["bmi"].isna().sum())),
            ("Smoking Unknown", int((g["smoking_status"] == "Unknown").sum()))
        ]
    ]
    def outlier_row(col):
        v = adults[col].dropna()
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        return {"factor": col, "observed_n": len(v), "lower_fence": lo, "upper_fence": hi,
                "flagged_n": int(((v < lo) | (v > hi)).sum()), "action": "Retained"}
    return {
        "data_audit": pd.DataFrame(audit_rows, columns=["measure", "value"]),
        "raw_missing_values": raw.isna().sum().reset_index(name="missing_count"),
        "raw_data_types": raw.dtypes.astype(str).reset_index(name="data_type"),
        "missingness_by_outcome": pd.DataFrame(missing_rows),
        "adult_outlier_audit": pd.DataFrame([outlier_row(c) for c in CONTINUOUS_COLUMNS])
    }
