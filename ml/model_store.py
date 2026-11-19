from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
from sklearn.pipeline import Pipeline

from core import config
from ml.model import DEFAULT_HYPERPARAMETERS


def save_churn_model(model: dict) -> None:
    path = config.MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            joblib.dump(model, file)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def load_churn_model() -> dict | None:
    try:
        model = joblib.load(config.MODEL_PATH)
    except FileNotFoundError:
        return None
    if not isinstance(model["pipeline"], Pipeline):
        raise ValueError("The saved model must contain a churn pipeline")
    model["training"].setdefault("model_type", "logreg")
    model["training"].setdefault("hyperparameters", DEFAULT_HYPERPARAMETERS["logreg"].copy())
    return model
