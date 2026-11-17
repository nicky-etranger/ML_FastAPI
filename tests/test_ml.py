import pandas as pd

from ml.dataset import FEATURE_COLUMNS, class_distribution, load_dataset
from ml.model import evaluate_churn_model, predict_churn, train_churn_model
from ml.preprocessing import prepare_churn_data, split_churn_data
from schemas.churn import FeatureVectorChurn


def test_prepare_and_split_churn_data(dataset_path):
    data = load_dataset()
    features, target = prepare_churn_data(data)
    assert list(features.columns) == FEATURE_COLUMNS
    assert pd.isna(features.loc[0, "usage_hours"])
    assert pd.isna(features.loc[0, "region"])
    assert target.name == "churn"

    train_x, test_x, train_y, test_y = split_churn_data(data)
    assert (len(train_x), len(test_x)) == (16, 4)
    assert train_x.index.intersection(test_x.index).empty
    assert class_distribution(train_y) == {0: 8, 1: 8}
    assert class_distribution(test_y) == {0: 2, 1: 2}
    assert test_x.equals(split_churn_data(data)[1])


def test_train_and_predict_churn(dataset_path, customer):
    train_x, test_x, train_y, test_y = split_churn_data(load_dataset())
    pipeline = train_churn_model(train_x, train_y)
    assert list(pipeline.feature_names_in_) == FEATURE_COLUMNS
    assert evaluate_churn_model(pipeline, test_x, test_y) == {"accuracy": 1, "f1": 1}
    rows = [FeatureVectorChurn(**{**customer, "region": "unseen", "failed_payments": value})
            for value in (0, 2)]
    assert [item.churn for item in predict_churn(pipeline, rows)] == [0, 1]
