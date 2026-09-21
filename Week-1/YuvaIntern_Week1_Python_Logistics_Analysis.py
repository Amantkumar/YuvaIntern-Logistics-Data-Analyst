"""
Yuva Intern - Logistics Data Analyst Internship
Week 1: Strategic Planning and Data Exploration in Logistics

Author: Aman Tiwari
Purpose:
    Illustrative Python implementation for the proposed logistics
    analytics workflow described in the Week 1 strategic planning report.

Note:
    This script is designed as a clean, professional implementation template.
    Replace the CSV path and column names with the actual dataset fields
    when a logistics dataset is selected.
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.cluster import KMeans


# ============================================================
# 2. LOAD DATA
# ============================================================

DATA_FILE = "logistics_data.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("LOGISTICS DATASET - INITIAL INSPECTION")
print("=" * 60)

print("\nFirst 5 records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())


# ============================================================
# 3. DATA CLEANING
# ============================================================

# Remove duplicate records
df = df.drop_duplicates().copy()

# Convert date column if available
if "delivery_date" in df.columns:
    df["delivery_date"] = pd.to_datetime(
        df["delivery_date"],
        errors="coerce"
    )

# Convert numerical fields if available
numeric_columns = [
    "distance_km",
    "shipment_volume",
    "delivery_time_hours",
    "transportation_cost"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

print("\nDataset after basic cleaning:")
print(df.head())


# ============================================================
# 4. KPI CALCULATIONS
# ============================================================

print("\n" + "=" * 60)
print("LOGISTICS KPI ANALYSIS")
print("=" * 60)

# On-Time Delivery Rate
if "on_time" in df.columns:
    on_time_rate = df["on_time"].mean() * 100
    print(f"On-Time Delivery Rate: {on_time_rate:.2f}%")

# Average Delivery Time
if "delivery_time_hours" in df.columns:
    avg_delivery_time = df["delivery_time_hours"].mean()
    print(
        f"Average Delivery Time: "
        f"{avg_delivery_time:.2f} hours"
    )

# Transportation Cost per Shipment
if "transportation_cost" in df.columns:
    cost_per_shipment = df["transportation_cost"].mean()
    print(
        f"Average Transportation Cost per Shipment: "
        f"{cost_per_shipment:.2f}"
    )

# Delivery Delay Rate
if "delayed" in df.columns:
    delay_rate = df["delayed"].mean() * 100
    print(f"Delivery Delay Rate: {delay_rate:.2f}%")


# ============================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe(include="all"))


# ============================================================
# 6. EXPLORATORY DATA ANALYSIS
# ============================================================

# Delivery Time Distribution
if "delivery_time_hours" in df.columns:

    plt.figure(figsize=(9, 5))

    sns.histplot(
        df["delivery_time_hours"].dropna(),
        kde=True
    )

    plt.title(
        "Delivery Time Distribution",
        fontsize=14,
        fontweight="bold"
    )
    plt.xlabel("Delivery Time (Hours)")
    plt.ylabel("Number of Deliveries")
    plt.tight_layout()
    plt.show()


# Transportation Cost Distribution
if "transportation_cost" in df.columns:

    plt.figure(figsize=(9, 5))

    sns.histplot(
        df["transportation_cost"].dropna(),
        kde=True
    )

    plt.title(
        "Transportation Cost Distribution",
        fontsize=14,
        fontweight="bold"
    )
    plt.xlabel("Transportation Cost")
    plt.ylabel("Number of Shipments")
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. CORRELATION ANALYSIS
# ============================================================

numeric_df = df.select_dtypes(include=np.number)

if not numeric_df.empty:

    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        linewidths=0.5
    )

    plt.title(
        "Logistics Variables Correlation Heatmap",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 8. DELIVERY TIME PREDICTION - REGRESSION
# ============================================================

required_features = [
    "distance_km",
    "shipment_volume"
]

target = "delivery_time_hours"

if all(
    column in df.columns
    for column in required_features + [target]
):

    model_data = df[
        required_features + [target]
    ].dropna()

    X = model_data[required_features]
    y = model_data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    regression_model = LinearRegression()

    regression_model.fit(
        X_train,
        y_train
    )

    predictions = regression_model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("DELIVERY TIME PREDICTION MODEL")
    print("=" * 60)

    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    # Actual vs Predicted
    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.7
    )

    plt.xlabel("Actual Delivery Time")
    plt.ylabel("Predicted Delivery Time")
    plt.title(
        "Actual vs Predicted Delivery Time",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 9. CLUSTERING - OPERATIONAL SEGMENTS
# ============================================================

cluster_features = [
    "distance_km",
    "delivery_time_hours"
]

if all(
    column in df.columns
    for column in cluster_features
):

    cluster_data = df[
        cluster_features
    ].dropna().copy()

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    cluster_data["cluster"] = kmeans.fit_predict(
        cluster_data[cluster_features]
    )

    print("\n" + "=" * 60)
    print("LOGISTICS OPERATIONAL CLUSTERS")
    print("=" * 60)

    print(
        cluster_data["cluster"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=cluster_data,
        x="distance_km",
        y="delivery_time_hours",
        hue="cluster",
        palette="deep",
        s=70
    )

    plt.title(
        "Delivery Segmentation Using K-Means",
        fontsize=14,
        fontweight="bold"
    )
    plt.xlabel("Distance (km)")
    plt.ylabel("Delivery Time (Hours)")
    plt.tight_layout()
    plt.show()


# ============================================================
# 10. BUSINESS INSIGHT FRAMEWORK
# ============================================================

print("\n" + "=" * 60)
print("BUSINESS INSIGHT FRAMEWORK")
print("=" * 60)

print("""
The analysis should be used to investigate:

1. Delivery reliability:
   Identify the proportion of shipments delivered on time.

2. Delivery efficiency:
   Understand average delivery duration and variation.

3. Cost efficiency:
   Monitor transportation cost per shipment.

4. Operational segments:
   Identify groups of deliveries with similar characteristics.

5. Predictive planning:
   Estimate delivery time using relevant operational variables.

6. Optimization opportunities:
   Use validated evidence to investigate route and resource
   allocation improvements.
""")


# ============================================================
# 11. CONCLUSION
# ============================================================

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

print("""
This Python workflow provides a structured foundation for
logistics data analysis.

The complete process includes:

Data Loading
    ↓
Data Cleaning
    ↓
KPI Calculation
    ↓
Exploratory Data Analysis
    ↓
Correlation Analysis
    ↓
Regression
    ↓
Clustering
    ↓
Business Insights
    ↓
Strategic Recommendations

Actual conclusions should only be reported after the selected
logistics dataset has been analyzed and the results validated.
""")
