import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from schemas.churn import PredictionResponseChurn


def train_churn_model(features: pd.DataFrame, target: pd.Series) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, NUMERIC_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
    ])
    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    return pipeline.fit(features, target)


def evaluate_churn_model(
    pipeline: Pipeline, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    predictions = pipeline.predict(features)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }


def predict_churn(pipeline: Pipeline, features: pd.DataFrame) -> list[PredictionResponseChurn]:
    labels = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)
    classes = [int(label) for label in pipeline.classes_]
    return [
        PredictionResponseChurn(
            churn=int(label),
            probabilities=dict(zip(classes, map(float, scores), strict=True)),
        )
        for label, scores in zip(labels, probabilities, strict=True)
    ]
