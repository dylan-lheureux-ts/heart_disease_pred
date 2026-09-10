import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drift_monitoring.targeted_drift import (
    run_targeted_drift
)
from drift_monitoring.drift_over_time import (
    run_drift_over_time
)
from drift_monitoring.detect_drift import (
    run_drift_detection
)



def main():

    print("=" * 60)
    print("PRODUCTION DRIFT MONITORING PIPELINE")
    print("=" * 60)

    # ========================================================
    # 1. Overall drift detection
    # ========================================================

    print("\n")
    print("=" * 60)
    print("STEP 1: OVERALL DATASET DRIFT")
    print("=" * 60)

    dataset_drift = run_drift_detection()

    # ========================================================
    # 2. Drift over time
    # ========================================================

    print("\n")
    print("=" * 60)
    print("STEP 2: DRIFT OVER TIME")
    print("=" * 60)

    run_drift_over_time()

    # ========================================================
    # 3. Targeted critical-feature drift
    # ========================================================

    print("\n")
    print("=" * 60)
    print("STEP 3: CRITICAL FEATURE DRIFT")
    print("=" * 60)

    run_targeted_drift()

    # ========================================================
    # 4. Final monitoring decision
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FINAL MONITORING STATUS")
    print("=" * 60)

    if dataset_drift:

        print(
            "\nDRIFT THRESHOLD EXCEEDED"
        )

        print(
            "Recommended action: "
            "Investigate drift and evaluate "
            "model performance."
        )

        print(
            "\nPipeline exiting with code 1."
        )

        sys.exit(1)

    else:

        print(
            "\nNO SIGNIFICANT DATASET DRIFT"
        )

        print(
            "Continue monitoring production data."
        )

        print(
            "\nPipeline exiting with code 0."
        )

        sys.exit(0)


if __name__ == "__main__":
    main()
