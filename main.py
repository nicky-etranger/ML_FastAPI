from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api import dataset, model, prediction
from ml.dataset import DatasetError
from ml.model import TrainingError
from ml.model_store import load_churn_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_churn_model()
    app.state.training_lock = Lock()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(dataset.router)
app.include_router(model.router)
app.include_router(prediction.router)


@app.exception_handler(DatasetError)
@app.exception_handler(TrainingError)
async def bad_training_data(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/")
def root():
    return {"message": "ml churn service is running"}
