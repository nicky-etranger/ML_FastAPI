from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib

from core import config


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
        return joblib.load(config.MODEL_PATH)
    except FileNotFoundError:
        return None
