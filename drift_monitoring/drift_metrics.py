import pandas as pd
import json
from evidently import Report
from evidently.presets import DataDriftPreset
from pathlib import Path


# Load the data
reports_dir = Path(__file__).resolve().parent / "reports"

reference = pd.read_csv(reports_dir / "reference_data.csv")
month1 = pd.read_csv(reports_dir / "month1_data.csv")
month2 = pd.read_csv(reports_dir / "month2_data.csv")
month3 = pd.read_csv(reports_dir / "month3_data.csv")


def get_drift_summary(reference, current, label):
    """Run drift detection and return a summary dictionary."""
    DATASET_DRIFT_THRESHOLD = 0.4  # 40% of features drifted indicates dataset drift

    # Exclude the target from feature drift monitoring
    feature_columns = [
        column for column in reference.columns
        if column != "num"
    ]

    reference_features = reference[feature_columns]
    current_features = current[feature_columns]

    report = Report(metrics=[DataDriftPreset()])

    snapshot = report.run(
        reference_data=reference_features,
        current_data=current_features
    )

    result = snapshot.dict()

    # First metric is DriftedColumnsCount
    drift_count_metric = result["metrics"][0]

    drifted_count = int(drift_count_metric["value"]["count"])
    drift_share = drifted_count / len(feature_columns)

    summary = {
        "period": label,
        "total_features": len(feature_columns),
        "drifted_features": drifted_count,
        "drift_share": round(drift_share, 3),
        "dataset_drift": drift_share >= DATASET_DRIFT_THRESHOLD,
    }

    # Extract per-feature drift details
    feature_metrics = result["metrics"][1:]

    feature_details = {}

    for metric in feature_metrics:
        column = metric["config"]["column"]
        threshold = metric["config"]["threshold"]
        drift_value = float(metric["value"])

        # For p-values:
        # p-value < 0.05 means statistically significant drift
        is_drifted = drift_value < threshold

        feature_details[column] = {
            "drifted": is_drifted,
            "threshold": threshold,
            "drift_score": round(drift_value, 4),
            "method": metric["config"]["method"],
        }

    summary["features"] = feature_details

    return summary


# Analyze all three months
for data, label in [(month1, "Month 1"), (month2, "Month 2"), (month3, "Month 3")]:
    summary = get_drift_summary(reference, data, label)

    print(f"\n{'=' * 60}")
    print(f"{label}: {summary['drifted_features']}/{summary['total_features']} features drifted "
          f"({summary['drift_share']*100:.1f}%)")
    print(f"Dataset drift detected: {summary['dataset_drift']}")
    print(f"{'=' * 60}")

    # Show which features drifted
    drifted = {k: v for k, v in summary["features"].items() if v["drifted"]}
    if drifted:
        print("Drifted features:")
        for feature, details in drifted.items():
            print(
                f"  {feature}: score = {details['drift_score']} (threshold = {details['threshold']})")
    else:
        print("No features showed significant drift.")
