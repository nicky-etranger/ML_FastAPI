from pydantic import BaseModel


class ValidationIssue(BaseModel):
    location: list[str | int]
    message: str
    type: str


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: list[ValidationIssue] | None = None
