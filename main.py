from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api import dataset, model
from ml.dataset import DatasetError
from schemas.churn import FeatureVectorChurn

app = FastAPI()
app.include_router(dataset.router)
app.include_router(model.router)


@app.exception_handler(DatasetError)
async def dataset_error(request: Request, exc: DatasetError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


@app.post("/predict")
def predict(features: FeatureVectorChurn) -> FeatureVectorChurn:
    return features
