import pandas as pd
import numpy as np
from pathlib import Path


def load_and_prepare():
    """
    Load the original heart disease dataset.

    Keeps numerical and categorical features in their original
    representation so drift can be simulated and monitored
    before model preprocessing.
    """

    url = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "heart_disease_uci.csv"
    )

    df = pd.read_csv(url).copy()

    # Remove rows with a missing target
    df = df.dropna(subset=["num"])

    # Convert the target to binary
    df["num"] = (df["num"] > 0).astype(int)

    # Remove columns not used by the model
    df = df.drop(columns=["id", "dataset"], errors="ignore")

    return df


def create_reference_and_production(df):
    """
    Split data into reference (training) and production batches.
    The reference set represents what the model was trained on.
    Production batches simulate data arriving over three months.
    """
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # First 60% is the reference (training) data
    split = int(len(df) * 0.6)
    reference = df.iloc[:split].copy()
    remaining = df.iloc[split:].copy()

    # Split remaining into three production "months"
    batch_size = len(remaining) // 3
    month1 = remaining.iloc[:batch_size].copy()
    month2 = remaining.iloc[batch_size:batch_size*2].copy()
    month3 = remaining.iloc[batch_size*2:].copy()

    return reference, month1, month2, month3


def introduce_drift(month2, month3):
    """
    Simulate realistic feature drift in months 2 and 3.

    Month 1 remains unchanged and represents normal production data.
    Month 2 introduces moderate population/measurement shifts.
    Month 3 introduces stronger but still plausible shifts.

    The target variable (num) is intentionally not modified because
    this simulation focuses on feature drift rather than concept drift.
    """

    # ============================================================
    # MONTH 2: MODERATE DRIFT
    # ============================================================

    # Cholesterol increases moderately.
    # Possible explanation: production population has a somewhat
    # higher prevalence of patients with elevated cholesterol.
    month2["chol"] = (
        month2["chol"]
        + np.random.normal(10, 5, len(month2))
    )
    month2["chol"] = month2["chol"].clip(100, 600)

    # Resting blood pressure increases slightly.
    # This represents a modest shift toward patients with higher
    # cardiovascular risk.
    month2["trestbps"] = (
        month2["trestbps"]
        + np.random.normal(5, 8, len(month2))
    )
    month2["trestbps"] = month2["trestbps"].clip(80, 220)

    # ============================================================
    # MONTH 3: SIGNIFICANT DRIFT
    # ============================================================

    # Cholesterol shifts further upward.
    month3["chol"] = (
        month3["chol"]
        + np.random.normal(25, 10, len(month3))
    )
    month3["chol"] = month3["chol"].clip(100, 650)

    # ST-segment depression (oldpeak) shifts upward.
    # This simulates a production population with slightly more
    # pronounced ST-segment depression during exercise.
    month3["oldpeak"] = (
        month3["oldpeak"]
        + np.random.normal(0.5, 0.25, len(month3))
    )
    month3["oldpeak"] = month3["oldpeak"].clip(0, 6)

    # Resting blood pressure shifts more substantially.
    month3["trestbps"] = (
        month3["trestbps"]
        + np.random.normal(12, 10, len(month3))
    )
    month3["trestbps"] = month3["trestbps"].clip(80, 230)

    # Age distribution shifts upward.
    # This simulates a production population containing more
    # older patients than the original training population.

    older_patients = np.random.randint(
        60,
        76,
        int(len(month3) * 0.25)
    )

    age_indices = np.random.choice(
        month3.index,
        size=len(older_patients),
        replace=False
    )

    month3.loc[age_indices, "age"] = older_patients

    # More patients experience exercise-induced angina.
    # The original column is binary: False = no, True = yes.

    month3["exang"] = month3["exang"].astype("boolean")

    angina_indices = np.random.choice(
        month3.index,
        size=int(len(month3) * 0.15),
        replace=False
    )

    month3.loc[angina_indices, "exang"] = True

    # Maximum heart rate shifts downward slightly.
    # This is consistent with a population containing more older
    # and higher-risk patients.

    month3["thalch"] = (
        month3["thalch"]
        - np.random.normal(8, 5, len(month3))
    )
    month3["thalch"] = month3["thalch"].clip(60, 220)

    return month2, month3


if __name__ == "__main__":
    print("Loading dataset...")
    df = load_and_prepare()
    print(f"Total rows: {len(df)}")

    print("\nSplitting into reference and production batches...")
    reference, month1, month2, month3 = create_reference_and_production(df)

    print("Introducing drift into months 2 and 3...")
    month2, month3 = introduce_drift(month2, month3)

    print(f"\nReference (training data): {len(reference)} rows")
    print(f"Month 1 (no drift):       {len(month1)} rows")
    print(f"Month 2 (moderate drift):  {len(month2)} rows")
    print(f"Month 3 (significant drift): {len(month3)} rows")

    # Save for use in other scripts
    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    reference.to_csv(reports_dir / "reference_data.csv", index=False)
    month1.to_csv(reports_dir / "month1_data.csv", index=False)
    month2.to_csv(reports_dir / "month2_data.csv", index=False)
    month3.to_csv(reports_dir / "month3_data.csv", index=False)

    print(f"\nData saved to: {reports_dir}")
    print("Ready for drift analysis.")
