import logging
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, Request

from api import dataset, model, prediction
from api.errors import add_error_handlers
from ml.dataset import DatasetError, load_dataset
from ml.model_store import load_churn_model

logger = logging.getLogger("uvicorn.error").getChild(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model = load_churn_model()
    except Exception as exc:
        logger.warning("Could not load the saved churn model: %s", exc)
        model = None
    if model is not None:
        logger.info("Loaded the saved churn model")
    app.state.model = model
    app.state.training_history = [model["training"]] if model else []
    app.state.training_lock = Lock()
    yield


app = FastAPI(lifespan=lifespan)
add_error_handlers(app)
app.include_router(dataset.router)
app.include_router(model.router)
app.include_router(prediction.router)


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


@app.get("/health")
def health(request: Request):
    try:
        load_dataset()
        dataset_loaded = True
    except DatasetError:
        dataset_loaded = False
    return {
        "status": "ok",
        "model_loaded": request.app.state.model is not None,
        "dataset_loaded": dataset_loaded,
    }
