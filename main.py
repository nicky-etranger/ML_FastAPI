from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api import dataset, model
from ml.dataset import DatasetError
from ml.model_store import load_churn_model
from schemas.churn import FeatureVectorChurn


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_churn_model()
    app.state.training_lock = Lock()
    yield


app = FastAPI(lifespan=lifespan)
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
