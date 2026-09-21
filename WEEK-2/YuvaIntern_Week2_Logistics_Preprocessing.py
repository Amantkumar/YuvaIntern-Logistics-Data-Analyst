"""
Yuva Intern - Week 2
Data Collection, Cleaning and Preprocessing for Logistics Analysis
Author: Aman Tiwari

This script demonstrates a reproducible preprocessing pipeline using the
Week 1 logistics dataset as the base for a controlled Week 2 simulation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

RAW_FILE = "logistics_data_raw_week2.csv"
CLEAN_FILE = "logistics_data_cleaned.csv"
PREPROCESSED_FILE = "logistics_data_preprocessed.csv"

# 1. Load data
df = pd.read_csv(RAW_FILE)

# 2. Initial inspection
print("Shape:", df.shape)
print(df.head())
print(df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# 3. Correct data types
df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")

numeric_cols = [
    "distance_km", "shipment_volume", "promised_time_hours",
    "delivery_time_hours", "transportation_cost"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. Remove exact duplicate records
df = df.drop_duplicates().reset_index(drop=True)

# 5. Handle missing numeric values with median imputation
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# 6. Standardize categorical values
for col in ["region", "vehicle_type", "traffic_level"]:
    df[col] = df[col].astype(str).str.strip().str.title()

# 7. Detect and cap outliers using the IQR rule
outlier_cols = [
    "distance_km", "shipment_volume",
    "delivery_time_hours", "transportation_cost"
]

for col in outlier_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[col] = df[col].clip(lower=lower, upper=upper)

# 8. Min-Max normalization
scaler = MinMaxScaler()
df[[c + "_normalized" for c in outlier_cols]] = scaler.fit_transform(
    df[outlier_cols]
)

# 9. Export outputs
df.to_csv(PREPROCESSED_FILE, index=False)

# Optional clean copy without normalized fields
base_cols = [
    "shipment_id", "delivery_date", "region", "vehicle_type",
    "traffic_level", "distance_km", "shipment_volume",
    "promised_time_hours", "delivery_time_hours",
    "transportation_cost", "on_time", "delayed"
]
df[base_cols].to_csv(CLEAN_FILE, index=False)

print("\nPreprocessing completed successfully.")
print("Final shape:", df.shape)
print("Remaining missing values:", df.isna().sum().sum())
