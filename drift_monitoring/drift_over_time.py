import pandas as pd
import numpy as np
import json
from evidently import Report
from evidently.metrics import ValueDrift
from pathlib import Path


# Load the data
reports_dir = Path(__file__).resolve().parent / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

reference = pd.read_csv(reports_dir / "reference_data.csv")
month1 = pd.read_csv(reports_dir / "month1_data.csv")
month2 = pd.read_csv(reports_dir / "month2_data.csv")
month3 = pd.read_csv(reports_dir / "month3_data.csv")

# Track a single important feature across all three months
feature_to_track = "age"

print(f"Tracking drift for '{feature_to_track}' over time")
print("=" * 60)

timeline = []

for data, label in [(month1, "Month 1"), (month2, "Month 2"), (month3, "Month 3")]:
    report = Report(metrics=[ValueDrift(column=feature_to_track)])
    snapshot = report.run(reference_data=reference, current_data=data)
    result = snapshot.dict()

    metric = result["metrics"][0]
    drift_score = float(metric["value"])
    threshold = metric["config"]["threshold"]

    entry = {
        "period": label,
        "ref_mean": round(reference[feature_to_track].mean(), 3),
        "current_mean": round(data[feature_to_track].mean(), 3),
        "ref_std": round(reference[feature_to_track].std(), 3),
        "current_std": round(data[feature_to_track].std(), 3),
        "P-value": round(drift_score, 6),
        "drift_detected": drift_score < threshold,
    }
    timeline.append(entry)

    status = "DRIFT" if entry["drift_detected"] else "OK"
    print(f"\n{label}:")
    print(
        f"  Reference mean: {entry['ref_mean']}  |  Current mean: {entry['current_mean']}")
    print(
        f"  Reference std:  {entry['ref_std']}  |  Current std:  {entry['current_std']}")
    print(f"  P-value:        {entry['P-value']}")
    print(f"  Status:         {status}")

# Save timeline for potential dashboard use
with open(reports_dir / "drift_timeline.json", "w") as f:
    json.dump(timeline, f, indent=2)

print(f"\nTimeline saved to {reports_dir / 'drift_timeline.json'}")
print("\nNotice how the drift score is monitored over time.")
print("Lower p-values indicate stronger statistical evidence of drift.")
