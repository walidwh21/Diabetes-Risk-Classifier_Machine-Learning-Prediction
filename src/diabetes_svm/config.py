"""Central place for paths, column names and random seeds."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "diabetes.csv"
MODEL_PATH = ROOT / "models" / "svm_pipeline.joblib"
ASSETS_DIR = ROOT / "assets"

TARGET = "Outcome"
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# In these columns a recorded 0 is physiologically impossible, so it really
# means "not measured". They are converted to NaN and imputed later.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
