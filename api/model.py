from datetime import datetime, timezone

from fastapi import APIRouter, Request

from ml.dataset import load_dataset
from ml.model import evaluate_churn_model, train_churn_model
from ml.model_store import save_churn_model
from ml.preprocessing import split_churn_data

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/train")
def train(request: Request):
    with request.app.state.training_lock:
        train_features, test_features, train_target, test_target = split_churn_data(load_dataset())
        pipeline = train_churn_model(train_features, train_target)
        metrics = evaluate_churn_model(pipeline, test_features, test_target)
        model = {
            "pipeline": pipeline,
            "training": {"trained_at": datetime.now(timezone.utc).isoformat(), "metrics": metrics},
        }
        save_churn_model(model)
        request.app.state.model = model
    return metrics


@router.get("/status")
def status(request: Request):
    model = request.app.state.model
    if model is None:
        return {"trained": False, "trained_at": None, "metrics": None}
    return {"trained": True, **model["training"]}
