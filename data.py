"""Loading, validating and time-splitting the PaySim dataset.

Dataset: https://www.kaggle.com/datasets/ealaxi/paysim1
Download it into ``data/`` (the folder is gitignored; never commit the raw file).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path("data/PS_20174392719_1491204439457_log.csv")

DTYPES: dict[str, str] = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}
REQUIRED_COLUMNS: list[str] = list(DTYPES)

# Dropped before modelling:
# - account IDs are random in PaySim, so they carry no real signal and are sensitive in real systems
# - isFlaggedFraud comes from the simulator's own rule, so it is label-adjacent and leaks the outcome
UNUSABLE_COLUMNS: list[str] = ["nameOrig", "nameDest", "isFlaggedFraud"]

# Default split points on the hourly ``step`` axis (PaySim spans steps 1-743).
DEFAULT_TRAIN_END = 600
DEFAULT_VAL_END = 680


@dataclass(frozen=True)
class SplitData:
    """Chronological train / validation / test partitions."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if any expected PaySim column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the PaySim CSV with explicit dtypes and validate its schema."""
    csv_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download PaySim from "
            "https://www.kaggle.com/datasets/ealaxi/paysim1 and place the CSV in data/."
        )
    df = pd.read_csv(csv_path, dtype=DTYPES)
    validate_columns(df)
    return df


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove ID columns and the label-adjacent isFlaggedFraud column."""
    return df.drop(columns=[c for c in UNUSABLE_COLUMNS if c in df.columns])


def time_split(
    df: pd.DataFrame,
    train_end: int = DEFAULT_TRAIN_END,
    val_end: int = DEFAULT_VAL_END,
) -> SplitData:
    """Split chronologically on ``step``: train <= train_end < val <= val_end < test.

    A time-based split avoids leaking future behaviour into training, which a
    random split would do.
    """
    if "step" not in df.columns:
        raise ValueError("DataFrame must contain a 'step' column")
    if not 0 < train_end < val_end:
        raise ValueError("Require 0 < train_end < val_end")
    if val_end >= df["step"].max():
        raise ValueError("val_end must be below the last step so the test set is not empty")

    train = df[df["step"] <= train_end]
    val = df[(df["step"] > train_end) & (df["step"] <= val_end)]
    test = df[df["step"] > val_end]
    return SplitData(train=train, val=val, test=test)


def summarize_split(split: SplitData) -> pd.DataFrame:
    """Rows, fraud counts and step ranges per partition, for sanity-checking a split."""
    rows = []
    for name, part in (("train", split.train), ("val", split.val), ("test", split.test)):
        frauds = int(part["isFraud"].sum()) if "isFraud" in part.columns else 0
        rows.append(
            {
                "split": name,
                "rows": len(part),
                "frauds": frauds,
                "fraud_rate": frauds / len(part) if len(part) else 0.0,
                "step_min": int(part["step"].min()) if len(part) else None,
                "step_max": int(part["step"].max()) if len(part) else None,
            }
        )
    return pd.DataFrame(rows).set_index("split")


def main() -> int:
    """`make data-check`: load the dataset and print the default split summary."""
    path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_raw(path)
    print(f"Loaded {len(df):,} rows; overall fraud rate {df['isFraud'].mean():.4%}")
    print(summarize_split(time_split(df)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
