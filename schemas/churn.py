from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

BinaryInt = Annotated[int, Field(ge=0, le=1)]


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: BinaryInt


class DatasetRowChurn(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    monthly_fee: float | None
    usage_hours: float | None
    support_requests: int | None
    account_age_months: int | None
    failed_payments: int | None
    region: str | None
    device_type: str | None
    payment_method: str | None
    autopay_enabled: BinaryInt | None
    churn: BinaryInt
