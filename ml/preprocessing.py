import pandas as pd
from sklearn.model_selection import train_test_split

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


def prepare_churn_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = data[FEATURE_COLUMNS].astype(object)
    features = features.where(features.notna(), float("nan"))
    return features, data["churn"]


def split_churn_data(data: pd.DataFrame):
    features, target = prepare_churn_data(data)
    try:
        split = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )
    except ValueError as exc:
        raise DatasetError("The dataset needs enough rows of both churn classes for an 80/20 split") from exc

    if split[2].nunique() != 2 or split[3].nunique() != 2:
        raise DatasetError("Both churn classes must be present in train and test")
    return split
