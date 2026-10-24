#!/usr/bin/env python
"""Command-line risk estimate.

Example:
    python scripts/predict.py --glucose 166 --bmi 25.8 --age 51 --pregnancies 5 \
        --blood-pressure 72 --skin-thickness 19 --insulin 175 --pedigree 0.587
Omit any option to treat that measurement as unknown.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diabetes_svm import predict_risk  # noqa: E402

OPTIONS = {
    "pregnancies": "Pregnancies",
    "glucose": "Glucose",
    "blood-pressure": "BloodPressure",
    "skin-thickness": "SkinThickness",
    "insulin": "Insulin",
    "bmi": "BMI",
    "pedigree": "DiabetesPedigreeFunction",
    "age": "Age",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in OPTIONS:
        parser.add_argument(f"--{flag}", type=float)
    args = parser.parse_args()
    values = {col: getattr(args, flag.replace("-", "_")) for flag, col in OPTIONS.items()}
    print(predict_risk(values))
    print("This is a statistical screening aid, not a medical diagnosis.")


if __name__ == "__main__":
    main()
