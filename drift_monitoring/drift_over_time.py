import pandas as pd
from evidently import Report
from evidently.metrics import ValueDrift
from pathlib import Path
import json


# ============================================================
# Configuration
# ============================================================

FEATURE_TO_TRACK = "age"

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


# ============================================================
# Drift over time
# ============================================================

def run_drift_over_time():

    reference = pd.read_csv(
        REPORTS_DIR / "reference_data.csv"
    )

    months = {
        "Month 1": pd.read_csv(
            REPORTS_DIR / "month1_data.csv"
        ),
        "Month 2": pd.read_csv(
            REPORTS_DIR / "month2_data.csv"
        ),
        "Month 3": pd.read_csv(
            REPORTS_DIR / "month3_data.csv"
        ),
    }

    timeline = []

    print("\nTracking feature:", FEATURE_TO_TRACK)

    for month_name, current_data in months.items():

        report = Report(
            metrics=[
                ValueDrift(
                    column=FEATURE_TO_TRACK
                )
            ]
        )

        snapshot = report.run(
            reference_data=reference,
            current_data=current_data
        )

        result = snapshot.dict()

        metric = result["metrics"][0]

        p_value = float(
            metric["value"]
        )

        reference_mean = reference[
            FEATURE_TO_TRACK
        ].mean()

        current_mean = current_data[
            FEATURE_TO_TRACK
        ].mean()

        drift_detected = p_value < 0.05

        status = (
            "DRIFT"
            if drift_detected
            else "OK"
        )

        print(
            f"\n{month_name}:"
        )

        print(
            f"  Reference mean: "
            f"{reference_mean:.3f}"
        )

        print(
            f"  Current mean:   "
            f"{current_mean:.3f}"
        )

        print(
            f"  P-value:        "
            f"{p_value:.6f}"
        )

        print(
            f"  Status:         "
            f"{status}"
        )

        timeline.append(
            {
                "month": month_name,
                "feature": FEATURE_TO_TRACK,
                "reference_mean": reference_mean,
                "current_mean": current_mean,
                "p_value": p_value,
                "drift_detected": drift_detected,
            }
        )

    # --------------------------------------------------------
    # Save timeline
    # --------------------------------------------------------

    output_file = (
        REPORTS_DIR
        / "drift_timeline.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            timeline,
            file,
            indent=4
        )

    print(
        f"\nTimeline saved to: "
        f"{output_file}"
    )

    print(
        "\nNotice how the age distribution "
        "changes over time."
    )

    print(
        "Lower p-values indicate stronger "
        "statistical evidence of drift."
    )

    return timeline


# ============================================================
# Allow standalone execution
# ============================================================

if __name__ == "__main__":
    run_drift_over_time()
