from typing import Annotated

from fastapi import APIRouter, Query

from ml.dataset import FEATURE_COLUMNS, class_distribution, load_dataset
from ml.preprocessing import split_churn_data

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/preview")
def preview(n: Annotated[int, Query(ge=1)] = 5):
    rows = load_dataset().head(n).astype(object)
    return rows.where(rows.notna(), None).to_dict(orient="records")


@router.get("/info")
def info():
    data = load_dataset()
    return {
        "rows": len(data),
        "columns": len(data.columns),
        "features": FEATURE_COLUMNS,
        "class_distribution": class_distribution(data["churn"]),
    }


@router.get("/split-info")
def split_info():
    _, _, train_target, test_target = split_churn_data(load_dataset())
    return {
        "train": {"size": len(train_target), "class_distribution": class_distribution(train_target)},
        "test": {"size": len(test_target), "class_distribution": class_distribution(test_target)},
    }
