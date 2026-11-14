from fastapi import APIRouter

from ml.dataset import load_dataset
from ml.model import evaluate_churn_model, train_churn_model
from ml.preprocessing import split_churn_data

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/train")
def train():
    train_features, test_features, train_target, test_target = split_churn_data(load_dataset())
    pipeline = train_churn_model(train_features, train_target)
    return evaluate_churn_model(pipeline, test_features, test_target)
