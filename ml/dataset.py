import csv

import pandas as pd
from pydantic import ValidationError

from core import config
from schemas.churn import DatasetRowChurn, FeatureVectorChurn

FEATURE_COLUMNS = list(FeatureVectorChurn.model_fields)
DATASET_COLUMNS = [*FEATURE_COLUMNS, "churn"]


class DatasetError(ValueError):
    pass


def load_dataset() -> pd.DataFrame:
    rows = []
    try:
        with config.DATASET_PATH.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, strict=True)
            if sorted(reader.fieldnames or []) != sorted(DATASET_COLUMNS):
                raise DatasetError("CSV header must contain the churn dataset columns")
            for row in reader:
                try:
                    rows.append(DatasetRowChurn.model_validate(row).model_dump())
                except ValidationError as exc:
                    raise DatasetError(f"Invalid CSV row at line {reader.line_num}") from exc
    except FileNotFoundError as exc:
        raise DatasetError("churn_dataset.csv was not found") from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetError("Could not read churn_dataset.csv") from exc

    if not rows:
        raise DatasetError("The churn dataset is empty")
    return pd.DataFrame(rows, columns=DATASET_COLUMNS)


def class_distribution(target: pd.Series) -> dict[int, int]:
    return {
        int(label): int(count)
        for label, count in target.value_counts().sort_index().items()
    }
