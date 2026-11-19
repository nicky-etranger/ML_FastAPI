import logging
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from ml.dataset import DatasetError
from ml.model import PredictionError, TrainingError
from schemas.errors import ErrorResponse

logger = logging.getLogger("uvicorn.error").getChild(__name__)

ERROR_CODES = {
    DatasetError: "invalid_dataset",
    TrainingError: "training_failed",
    PredictionError: "prediction_failed",
}


def error_body(code: str, message: str, details: list[dict] | None = None) -> dict:
    return {"code": code, "message": message, "details": details}


def error_example(code: str, message: str, details: list[dict] | None = None) -> dict:
    return {
        "model": ErrorResponse,
        "content": {"application/json": {"example": error_body(code, message, details)}},
    }


async def http_error(request: Request, exc: HTTPException):
    code = HTTPStatus(exc.status_code).name.lower()
    logger.warning("%s %s: %s", request.method, request.url.path, code)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(code, exc.detail),
        headers=exc.headers,
    )


async def invalid_request(request: Request, exc: RequestValidationError):
    logger.warning("%s %s: validation_error", request.method, request.url.path)
    details = [
        {"location": issue["loc"], "message": issue["msg"], "type": issue["type"]}
        for issue in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=error_body("validation_error", "Invalid request data", details),
    )


async def ml_error(request: Request, exc: ValueError):
    code = ERROR_CODES[type(exc)]
    logger.warning("%s %s: %s", request.method, request.url.path, code)
    return JSONResponse(status_code=400, content=error_body(code, str(exc)))


async def internal_error(request: Request, exc: Exception):
    logger.error("%s %s: %s", request.method, request.url.path, type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content=error_body("internal_server_error", "An internal server error occurred"),
    )


def add_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(HTTPException, http_error)
    app.add_exception_handler(RequestValidationError, invalid_request)
    for error in ERROR_CODES:
        app.add_exception_handler(error, ml_error)
    app.add_exception_handler(Exception, internal_error)
