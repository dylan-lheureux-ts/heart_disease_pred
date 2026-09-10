import pandas as pd
from evidently import Report
from evidently.metrics import ValueDrift
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

CRITICAL_FEATURES = [
    "sex",
    "ca",
    "chol",
    "thalch",
    "age",
]

P_VALUE_THRESHOLD = 0.05


# ============================================================
# Targeted drift analysis
# ============================================================

def run_targeted_drift():

    reference = pd.read_csv(
        REPORTS_DIR / "reference_data.csv"
    )

    month3 = pd.read_csv(
        REPORTS_DIR / "month3_data.csv"
    )

    metrics = [
        ValueDrift(column=feature)
        for feature in CRITICAL_FEATURES
    ]

    report = Report(
        metrics=metrics
    )

    snapshot = report.run(
        reference_data=reference,
        current_data=month3
    )

    result = snapshot.dict()

    print(
        "\nCritical Feature Drift Analysis "
        "(Month 3)"
    )

    print("=" * 60)

    drift_detected = False

    for i, feature in enumerate(
        CRITICAL_FEATURES
    ):

        metric = result["metrics"][i]

        p_value = float(
            metric["value"]
        )

        threshold = metric[
            "config"
        ]["threshold"]

        method = metric[
            "config"
        ]["method"]

        # IMPORTANT:
        # ValueDrift returns a p-value.
        # Lower p-values indicate stronger evidence
        # of drift.

        drifted = p_value < threshold

        if drifted:
            drift_detected = True

        status = (
            "DRIFT DETECTED"
            if drifted
            else "STABLE"
        )

        print(
            f"\n{feature}:"
        )

        print(
            f"  Status:    {status}"
        )

        print(
            f"  P-value:   {p_value:.6f}"
        )

        print(
            f"  Threshold: {threshold}"
        )

        print(
            f"  Test used: {method}"
        )

    # --------------------------------------------------------
    # Save HTML report
    # --------------------------------------------------------

    report_file = (
        REPORTS_DIR
        / "critical_features_month3.html"
    )

    snapshot.save_html(
        str(report_file)
    )

    print(
        f"\nDetailed report: "
        f"{report_file}"
    )

    return drift_detected


# ============================================================
# Allow standalone execution
# ============================================================

if __name__ == "__main__":
    run_targeted_drift()
