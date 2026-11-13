from typing import Annotated

from fastapi import APIRouter, Query

from ml.dataset import FEATURE_COLUMNS, class_distribution, load_dataset

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/preview")
def preview(n: Annotated[int, Query(ge=1)] = 5):
    return load_dataset().head(n).to_dict(orient="records")


@router.get("/info")
def info():
    data = load_dataset()
    return {
        "rows": len(data),
        "columns": len(data.columns),
        "features": FEATURE_COLUMNS,
        "class_distribution": class_distribution(data["churn"]),
    }
