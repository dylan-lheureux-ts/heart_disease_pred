from pathlib import Path
from evidently.presets import DataDriftPreset
from evidently import Report
import pandas as pd


# ============================================================
# Configuration
# ============================================================

DRIFT_THRESHOLD = 0.40
FEATURE_P_VALUE_THRESHOLD = 0.05

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


# ============================================================
# Overall drift detection
# ============================================================

def run_drift_detection():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Reports directory:", REPORTS_DIR)
    print(
        "Reports directory exists:",
        REPORTS_DIR.exists()
    )

    reference = pd.read_csv(
        REPORTS_DIR / "reference_data.csv"
    )

    month1 = pd.read_csv(
        REPORTS_DIR / "month1_data.csv"
    )

    month2 = pd.read_csv(
        REPORTS_DIR / "month2_data.csv"
    )

    month3 = pd.read_csv(
        REPORTS_DIR / "month3_data.csv"
    )

    # --------------------------------------------------------
    # Features to monitor
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in reference.columns
        if column != "num"
    ]

    reference_features = reference[
        feature_columns
    ]

    # --------------------------------------------------------
    # Analyze each month
    # --------------------------------------------------------

    dataset_drift_detected = False

    for month_name, current_data in [
        ("Month 1", month1),
        ("Month 2", month2),
        ("Month 3", month3),
    ]:

        print("\n" + "=" * 60)

        if month_name == "Month 1":
            print(
                "DRIFT REPORT: Month 1 "
                "(expected: no drift)"
            )

        elif month_name == "Month 2":
            print(
                "DRIFT REPORT: Month 2 "
                "(expected: moderate drift)"
            )

        else:
            print(
                "DRIFT REPORT: Month 3 "
                "(expected: significant drift)"
            )

        print("=" * 60)

        # ----------------------------------------------------
        # Run Evidently
        # ----------------------------------------------------

        report = Report(
            metrics=[
                DataDriftPreset()
            ]
        )

        snapshot = report.run(
            reference_data=reference_features,
            current_data=current_data[
                feature_columns
            ]
        )

        result = snapshot.dict()

        # ----------------------------------------------------
        # Find drift share
        # ----------------------------------------------------

        drift_share = None

        for metric in result["metrics"]:

            metric_name = metric.get(
                "metric_name",
                ""
            )

            if "DriftedColumnsShare" in metric_name:

                drift_share = float(
                    metric["value"]
                )

                break

        # ----------------------------------------------------
        # Calculate drift share if needed
        # ----------------------------------------------------

        if drift_share is None:

            drifted_count = 0

            for metric in result["metrics"]:

                metric_name = metric.get(
                    "metric_name",
                    ""
                )

                if "ValueDrift" in metric_name:

                    value = metric.get(
                        "value"
                    )

                    if value is not None:

                        p_value = float(value)

                        if (
                            p_value
                            < FEATURE_P_VALUE_THRESHOLD
                        ):
                            drifted_count += 1

            drift_share = (
                drifted_count
                / len(feature_columns)
            )

        drifted_count = round(
            drift_share
            * len(feature_columns)
        )

        dataset_drift = (
            drift_share
            >= DRIFT_THRESHOLD
        )

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print(
            f"\n{month_name}: "
            f"{drifted_count}/"
            f"{len(feature_columns)} "
            f"features drifted "
            f"({drift_share:.1%})"
        )

        print(
            f"Dataset drift detected: "
            f"{dataset_drift}"
        )

        # ----------------------------------------------------
        # Print drifted features
        # ----------------------------------------------------
        drifted_features = []

        for metric in result["metrics"]:
            metric_name = metric.get(
                "metric_name",
                ""
            )

            value = metric.get("value")

            if (
                "ValueDrift" in metric_name
                and value is not None
            ):
                p_value = float(value)

                if p_value < FEATURE_P_VALUE_THRESHOLD:
                    feature = metric.get(
                        "config",
                        {}
                    ).get("column")

                    if feature is not None:
                        drifted_features.append(
                            (
                                feature,
                                p_value
                            )
                        )

        if drifted_features:
            print(
                "Drifted features:"
            )

            for feature, p_value in drifted_features:
                print(
                    f"  {feature}: "
                    f"p-value = "
                    f"{p_value:.6f}"
                )
        else:
            print(
                "No features showed "
                "significant drift."
            )



        # ----------------------------------------------------
        # Save HTML report
        # ----------------------------------------------------

        report_file = (
            REPORTS_DIR
            / f"drift_{month_name.lower().replace(' ', '')}.html"
        )

        snapshot.save_html(
            str(report_file)
        )

        print(
            "Report saved to:",
            report_file
        )

        if dataset_drift:

            dataset_drift_detected = True

    # --------------------------------------------------------
    # Return final result to monitor_drift.py
    # --------------------------------------------------------

    return dataset_drift_detected


# ============================================================
# Allow standalone execution
# ============================================================

if __name__ == "__main__":

    drift_detected = run_drift_detection()

    if drift_detected:

        print(
            "\nOverall drift threshold exceeded."
        )

    else:

        print(
            "\nOverall drift remained "
            "below the threshold."
        )
