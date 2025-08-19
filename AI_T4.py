"""
AI_T4.py
Data Handling, Cleaning, Visualization, and Feature Engineering
Dataset: No-show appointments (Kaggle)
"""

# =========================
# 📦 Step 1: Import Libraries
# =========================
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio

# Make plots look nice
sns.set(style="whitegrid")
pio.renderers.default = "browser"  # opens interactive plots in browser


# =========================
# 📂 Step 2: Load Dataset
# =========================
def load_dataset(path="data/KaggleV2-May-2016.csv"):
    """Load dataset from CSV file."""
    if not os.path.exists(path):
        print(f"❌ Dataset not found at {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"✅ Dataset loaded. Shape: {df.shape}")
    return df


# =========================
# 🧹 Step 3: Data Cleaning
# =========================
def parse_dates_and_target(df):
    """Convert date columns to datetime and compute waiting days + target."""
    # Ensure lowercase column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Convert to datetime
    df["scheduledday"] = pd.to_datetime(df["scheduledday"], errors="coerce")
    df["appointmentday"] = pd.to_datetime(df["appointmentday"], errors="coerce")

    # Compute waiting days (FIXED: removed .dt.date)
    df["waiting_days"] = (df["appointmentday"] - df["scheduledday"]).dt.days

    # Target column
    df["no_show"] = df["no-show"].map({"No": 0, "Yes": 1})

    return df


def clean_data(df):
    """Basic cleaning: handle missing values, drop duplicates, fix negatives."""
    print("\n🔹 Missing values before cleaning:")
    print(df.isnull().sum())

    # Drop duplicates
    df = df.drop_duplicates()

    # Remove impossible ages
    df = df[(df["age"] >= 0) & (df["age"] <= 115)]

    # Handle negative waiting days (appointments in past)
    df = df[df["waiting_days"] >= 0]

    print(f"✅ Cleaned dataset. Shape: {df.shape}")
    return df


# =========================
# 📊 Step 4: Visualization
# =========================
def visualize_data(df):
    """Generate different visualizations using Matplotlib, Seaborn, Plotly."""

    # ---- Matplotlib ----
    plt.figure(figsize=(8, 5))
    df["age"].hist(bins=30)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()

    # ---- Seaborn ----
    plt.figure(figsize=(8, 5))
    sns.countplot(x="no_show", data=df, palette="Set2")
    plt.title("No-Show Distribution")
    plt.show()

    # ---- Plotly ----
    fig = px.histogram(df, x="waiting_days", color="no_show", barmode="overlay",
                       title="Waiting Days vs No-Show")
    fig.show()

    fig2 = px.box(df, x="no_show", y="age", title="Age vs No-Show")
    fig2.show()


# =========================
# ⚙️ Step 5: Feature Engineering
# =========================
def feature_engineering(df):
    """Create useful features from existing columns."""

    # Extract day of week from appointment
    df["appointment_dow"] = df["appointmentday"].dt.dayofweek  # 0=Mon

    # Is weekend
    df["is_weekend"] = df["appointment_dow"].isin([5, 6]).astype(int)

    # Waiting time buckets
    df["waiting_bucket"] = pd.cut(df["waiting_days"],
                                  bins=[-1, 0, 7, 30, 90, 365],
                                  labels=["same_day", "within_week", "within_month", "within_3mo", "long_wait"])

    # Age group
    df["age_group"] = pd.cut(df["age"],
                             bins=[0, 12, 18, 40, 60, 120],
                             labels=["child", "teen", "adult", "middle_age", "senior"])

    print("✅ Feature engineering done. New columns added:")
    print(["appointment_dow", "is_weekend", "waiting_bucket", "age_group"])
    return df


# =========================
# 📈 Step 6: Correlation & Summary
# =========================
def correlation_analysis(df):
    """Check correlations between numeric features and target."""
    plt.figure(figsize=(10, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()


# =========================
# 🚀 Main Function
# =========================
def main():
    print("\n══════════════════════════════════════════════════════════════════════")
    print("🔹 Loading dataset")
    print("══════════════════════════════════════════════════════════════════════")

    df = load_dataset("KaggleV2-May-2016.csv")  # file in same folder or adjust path

    print("\n══════════════════════════════════════════════════════════════════════")
    print("🔹 Cleaning & manipulating data")
    print("══════════════════════════════════════════════════════════════════════")
    df = parse_dates_and_target(df)
    df = clean_data(df)

    print("\n══════════════════════════════════════════════════════════════════════")
    print("🔹 Visualization")
    print("══════════════════════════════════════════════════════════════════════")
    visualize_data(df)

    print("\n══════════════════════════════════════════════════════════════════════")
    print("🔹 Feature Engineering")
    print("══════════════════════════════════════════════════════════════════════")
    df = feature_engineering(df)

    print("\n══════════════════════════════════════════════════════════════════════")
    print("🔹 Correlation Analysis")
    print("══════════════════════════════════════════════════════════════════════")
    correlation_analysis(df)

    print("\n✅ All steps completed successfully!")


if __name__ == "__main__":
    main()

