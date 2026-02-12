import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ==========================================================
# 1️⃣ Create Synthetic Readmission Target (Logical Approach)
# ==========================================================

def create_readmission_target(df):
    """
    Create synthetic readmission risk based on:
    - High length of stay
    - High total cost
    - Emergency admission
    """

    # Fill missing values temporarily for calculation
    df["length_of_stay"] = df["length_of_stay"].fillna(df["length_of_stay"].median())
    df["total_costs"] = df["total_costs"].fillna(df["total_costs"].median())

    # Define thresholds
    los_threshold = df["length_of_stay"].quantile(0.75)
    cost_threshold = df["total_costs"].quantile(0.75)

    # Admission type safety check
    if "admission_type" in df.columns:
        emergency_flag = df["admission_type"].astype(str).str.lower().str.contains("emergency")
    else:
        emergency_flag = False

    # Risk logic
    df["readmission"] = (
        (df["length_of_stay"] > los_threshold) |
        (df["total_costs"] > cost_threshold) |
        (emergency_flag)
    ).astype(int)

    return df


# ==========================================================
# 2️⃣ Load & Prepare Data
# ==========================================================

def load_and_prepare_data():

    df = pd.read_csv("data/hip_replacement.csv")

    # --------------------------------------------------
    # Remove ID / Leakage Columns
    # --------------------------------------------------

    columns_to_remove = [
        "operating_certificate_number",
        "facility_id",
        "facility_name",
        "attending_provider_license_number",
        "operating_provider_license_number"
    ]

    df = df.drop(columns=columns_to_remove, errors="ignore")

    # --------------------------------------------------
    # Create Readmission Target
    # --------------------------------------------------

    df = create_readmission_target(df)

    # --------------------------------------------------
    # Define Targets
    # --------------------------------------------------

    y_los = df["length_of_stay"]
    y_cost = df["total_costs"]
    y_readmission = df["readmission"]

    # --------------------------------------------------
    # Remove Leakage Columns from Features
    # --------------------------------------------------

    leakage_columns = [
        "length_of_stay",
        "total_costs",
        "readmission",
        "patient_disposition"  # prevent leakage
    ]

    X = df.drop(columns=leakage_columns, errors="ignore")

    # --------------------------------------------------
    # Train-Test Split (ONLY ONCE)
    # --------------------------------------------------

    X_train, X_test, y_read_train, y_read_test = train_test_split(
        X,
        y_readmission,
        test_size=0.2,
        random_state=42,
        stratify=y_readmission
    )

    # Use same indices for other targets
    y_los_train = y_los.loc[X_train.index]
    y_los_test = y_los.loc[X_test.index]

    y_cost_train = y_cost.loc[X_train.index]
    y_cost_test = y_cost.loc[X_test.index]

    # --------------------------------------------------
    # Feature Types
    # --------------------------------------------------

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # --------------------------------------------------
    # Preprocessing Pipelines
    # --------------------------------------------------

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )

    return (
        X_train, X_test,
        y_los_train, y_los_test,
        y_cost_train, y_cost_test,
        y_read_train, y_read_test,
        preprocessor
    )
