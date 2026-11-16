import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.dataset import FEATURE_COLUMNS, DatasetError

NUMERIC_COLUMNS = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]
CATEGORICAL_COLUMNS = ["region", "device_type", "payment_method"]


def prepare_churn_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data[FEATURE_COLUMNS].astype(object)
    return features.where(features.notna(), float("nan"))


def prepare_churn_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return prepare_churn_features(data), data["churn"]


def split_churn_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features, target = prepare_churn_data(data)
    try:
        train_x, test_x, train_y, test_y = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )
    except ValueError as exc:
        raise DatasetError("The dataset needs enough rows of both churn classes for an 80/20 split") from exc

    if train_y.nunique() != 2 or test_y.nunique() != 2:
        raise DatasetError("Both churn classes must be present in train and test")
    return train_x, test_x, train_y, test_y


def churn_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
    ])
