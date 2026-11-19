import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from ml.preprocessing import churn_preprocessor, prepare_churn_features
from schemas.churn import FeatureVectorChurn, PredictionResponseChurn, TrainingConfigChurn

CLASSIFIERS = {
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
}
DEFAULT_HYPERPARAMETERS = {
    "logreg": {"max_iter": 1000},
    "random_forest": {"random_state": 42},
}


class TrainingError(ValueError):
    pass


class PredictionError(ValueError):
    pass


def train_churn_model(
    features: pd.DataFrame, target: pd.Series, config: TrainingConfigChurn | None = None
) -> Pipeline:
    config = config or TrainingConfigChurn()
    parameters = DEFAULT_HYPERPARAMETERS[config.model_type] | config.hyperparameters
    try:
        classifier = CLASSIFIERS[config.model_type](**parameters)
        pipeline = Pipeline([("preprocessing", churn_preprocessor()), ("classifier", classifier)])
        return pipeline.fit(features, target)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingError(f"Could not train the model: {exc}") from exc


def evaluate_churn_model(
    pipeline: Pipeline, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    predictions = pipeline.predict(features)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }


def predict_churn(
    pipeline: Pipeline, rows: list[FeatureVectorChurn]
) -> list[PredictionResponseChurn]:
    try:
        features = prepare_churn_features(pd.DataFrame([row.model_dump() for row in rows]))
        labels = pipeline.predict(features)
        probabilities = pipeline.predict_proba(features)
    except (ValueError, TypeError) as exc:
        raise PredictionError("Could not predict churn for the supplied features") from exc
    classes = [int(label) for label in pipeline.classes_]
    return [
        PredictionResponseChurn(
            churn=int(label),
            probabilities=dict(zip(classes, map(float, scores), strict=True)),
        )
        for label, scores in zip(labels, probabilities, strict=True)
    ]
