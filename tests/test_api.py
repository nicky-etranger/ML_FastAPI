import pytest
from fastapi.testclient import TestClient

from main import app
from ml.dataset import DATASET_COLUMNS, FEATURE_COLUMNS


def test_training_prediction_and_restart(dataset_path, model_path, customer):
    with TestClient(app) as client:
        preview = client.get("/dataset/preview?n=1").json()
        assert preview[0]["usage_hours"] is None
        assert client.get("/dataset/info").json()["rows"] == 20
        assert client.get("/dataset/split-info").json()["test"]["size"] == 4
        assert [field["name"] for field in client.get("/model/schema").json()["features"]] == FEATURE_COLUMNS

        assert client.post("/model/train").status_code == 200
        response = client.post("/model/train", json={
            "model_type": "random_forest", "hyperparameters": {"n_estimators": 10},
        })
        assert response.status_code == 200
        status = client.get("/model/status").json()
        assert status["trained"] and status["trained_at"]
        assert status["model_type"] == "random_forest"
        assert status["hyperparameters"]["n_estimators"] == 10
        assert status["metrics"] == response.json()
        history = client.get("/model/metrics").json()
        assert [run["model_type"] for run in history["history"]] == ["logreg", "random_forest"]
        assert history["latest"]["metrics"] == response.json()
        assert model_path.is_file()

        response = client.post("/predict", json=customer)
        assert response.status_code == 200
        prediction = response.json()
        assert prediction["churn"] in (0, 1)
        assert set(prediction["probabilities"]) == {"0", "1"}
        assert sum(prediction["probabilities"].values()) == pytest.approx(1)
        batch = client.post("/predict", json=[customer, {**customer, "failed_payments": 2}])
        assert batch.status_code == 200
        assert batch.json()[0] == prediction and len(batch.json()) == 2

    with TestClient(app) as client:
        assert client.get("/model/status").json() == status
        assert client.post("/predict", json=customer).json() == prediction


def test_prediction_requires_a_model(customer):
    with TestClient(app) as client:
        response = client.post("/predict", json=customer)
        assert response.status_code == 503
        assert response.json()["code"] == "service_unavailable"


def test_predict_validates_features(customer):
    invalid_rows = [
        [],
        {key: value for key, value in customer.items() if key != "region"},
        {**customer, "unexpected": "extra"},
        {**customer, "monthly_fee": "wrong"},
    ]
    with TestClient(app) as client:
        for row in invalid_rows:
            response = client.post("/predict", json=row)
            assert response.status_code == 422
            assert response.json()["code"] == "validation_error"
            assert response.json()["details"]


def test_training_rejects_invalid_dataset(dataset_path):
    dataset_path.write_text(",".join(DATASET_COLUMNS) + "\n", encoding="utf-8")
    with TestClient(app) as client:
        response = client.post("/model/train")
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_dataset"
        assert "empty" in response.json()["message"]
