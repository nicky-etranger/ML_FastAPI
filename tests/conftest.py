import csv

import pytest

from core import config
from ml.dataset import DATASET_COLUMNS


@pytest.fixture
def customer():
    return {
        "monthly_fee": 19.99,
        "usage_hours": 20,
        "support_requests": 1,
        "account_age_months": 12,
        "failed_payments": 0,
        "region": "europe",
        "device_type": "mobile",
        "payment_method": "card",
        "autopay_enabled": 1,
    }


@pytest.fixture(autouse=True)
def model_path(monkeypatch, tmp_path):
    path = tmp_path / "models" / "churn_model.joblib"
    monkeypatch.setattr(config, "MODEL_PATH", path)
    return path


@pytest.fixture
def dataset_path(monkeypatch, tmp_path, customer):
    path = tmp_path / "churn_dataset.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        for index in range(20):
            churn = index % 2
            writer.writerow({
                **customer,
                "usage_hours": None if index == 0 else index,
                "region": None if index == 0 else customer["region"],
                "failed_payments": churn * 2,
                "churn": churn,
            })

    monkeypatch.setattr(config, "DATASET_PATH", path)
    return path
