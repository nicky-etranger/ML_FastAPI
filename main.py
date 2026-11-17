from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI

from api import dataset, model, prediction
from api.errors import add_error_handlers
from ml.model_store import load_churn_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = load_churn_model()
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
