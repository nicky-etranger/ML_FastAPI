from datetime import datetime, timezone

from fastapi import APIRouter, Request

from api.errors import error_body
from ml.dataset import FEATURE_COLUMNS, load_dataset
from ml.model import DEFAULT_HYPERPARAMETERS, evaluate_churn_model, train_churn_model
from ml.model_store import save_churn_model
from ml.preprocessing import split_churn_data
from schemas.churn import FeatureVectorChurn, TrainingConfigChurn

from schemas.errors import ErrorResponse

router = APIRouter(prefix="/model", tags=["model"])


@router.post(
    "/train",
    responses={
        400: {
            "model": ErrorResponse,
            "content": {"application/json": {"examples": {
                "empty_dataset": {"value": error_body("invalid_dataset", "The churn dataset is empty")},
                "training_failed": {"value": error_body("training_failed", "Could not train the model: invalid hyperparameters")},
            }}},
        },
        422: {"model": ErrorResponse},
    },
)
def train(request: Request, config: TrainingConfigChurn | None = None):
    config = config or TrainingConfigChurn()
    with request.app.state.training_lock:
        train_features, test_features, train_target, test_target = split_churn_data(load_dataset())
        pipeline = train_churn_model(train_features, train_target, config)
        metrics = evaluate_churn_model(pipeline, test_features, test_target)
        model = {
            "pipeline": pipeline,
            "training": {
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "model_type": config.model_type,
                "hyperparameters": DEFAULT_HYPERPARAMETERS[config.model_type] | config.hyperparameters,
                "metrics": metrics,
            },
        }
        save_churn_model(model)
        request.app.state.model = model
        request.app.state.training_history.append(model["training"])
    return metrics


@router.get("/status")
def status(request: Request):
    model = request.app.state.model
    if model is None:
        return {
            "trained": False,
            "trained_at": None,
            "model_type": None,
            "hyperparameters": None,
            "metrics": None,
        }
    return {"trained": True, **model["training"]}


@router.get("/schema")
def schema():
    properties = FeatureVectorChurn.model_json_schema()["properties"]
    return {"features": [{"name": name, "type": properties[name]["type"]} for name in FEATURE_COLUMNS]}


@router.get("/metrics")
def metrics(request: Request):
    history = request.app.state.training_history[-5:]
    return {"latest": history[-1] if history else None, "history": history}
