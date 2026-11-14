from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "churn_dataset.csv"
MODEL_PATH = PROJECT_DIR / "models" / "churn_model.joblib"
