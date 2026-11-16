from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI

from api import dataset, model, prediction
from api.errors import add_error_handlers
from ml.model_store import load_churn_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_churn_model()
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
