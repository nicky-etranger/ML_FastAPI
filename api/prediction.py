from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import Field

from api.errors import error_body
from ml.model import predict_churn
from schemas.churn import FeatureVectorChurn, PredictionResponseChurn

from schemas.errors import ErrorResponse

router = APIRouter(tags=["model"])

FEATURE_EXAMPLE = {
    "monthly_fee": 19.99,
    "usage_hours": 21.48,
    "support_requests": 2,
    "account_age_months": 1,
    "failed_payments": 0,
    "region": "america",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1,
}
PREDICTION_EXAMPLE = {"churn": 0, "probabilities": {"0": 0.8, "1": 0.2}}
MODEL_UNAVAILABLE = "Train the churn model with POST /model/train first"
FeatureBatch = Annotated[list[FeatureVectorChurn], Field(min_length=1)]


@router.post(
    "/predict",
    description="Send one customer or a non-empty list. The response matches the input shape.",
    responses={
        200: {
            "content": {
                "application/json": {
                    "examples": {
                        "single": {"value": PREDICTION_EXAMPLE},
                        "batch": {"value": [PREDICTION_EXAMPLE]},
                    }
                }
            }
        },
        400: {"model": ErrorResponse},
        422: {
            "model": ErrorResponse,
            "content": {"application/json": {"example": error_body(
                "validation_error",
                "Invalid request data",
                [{"location": ["body", "FeatureVectorChurn", "usage_hours"],
                  "message": "Input should be a valid number", "type": "float_type"}],
            )}},
        },
        503: {
            "model": ErrorResponse,
            "content": {"application/json": {"example": error_body(
                "service_unavailable", MODEL_UNAVAILABLE
            )}},
        },
    },
)
def predict(
    request: Request,
    features: Annotated[
        FeatureVectorChurn | FeatureBatch,
        Body(openapi_examples={
            "single": {"value": FEATURE_EXAMPLE},
            "batch": {"value": [FEATURE_EXAMPLE]},
        }),
    ],
) -> PredictionResponseChurn | list[PredictionResponseChurn]:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE)

    batch = isinstance(features, list)
    rows = features if batch else [features]
    results = predict_churn(model["pipeline"], rows)
    return results if batch else results[0]
