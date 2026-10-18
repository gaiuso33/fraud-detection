import pandas as pd
import pytest

from fraud.data import (
    REQUIRED_COLUMNS,
    drop_unusable_columns,
    summarize_split,
    time_split,
    validate_columns,
)


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    n = 20
    return pd.DataFrame(
        {
            "step": list(range(1, n + 1)),
            "type": ["TRANSFER"] * n,
            "amount": [100.0] * n,
            "nameOrig": [f"C{i}" for i in range(n)],
            "oldbalanceOrg": [500.0] * n,
            "newbalanceOrig": [400.0] * n,
            "nameDest": [f"M{i}" for i in range(n)],
            "oldbalanceDest": [0.0] * n,
            "newbalanceDest": [100.0] * n,
            "isFraud": [0] * (n - 2) + [1, 1],
            "isFlaggedFraud": [0] * n,
        }
    )


def test_validate_columns_accepts_full_schema(tiny_df):
    validate_columns(tiny_df)


def test_validate_columns_reports_missing(tiny_df):
    with pytest.raises(ValueError, match="amount"):
        validate_columns(tiny_df.drop(columns=["amount"]))


def test_drop_unusable_columns_removes_ids_and_flag(tiny_df):
    out = drop_unusable_columns(tiny_df)
    assert not {"nameOrig", "nameDest", "isFlaggedFraud"} & set(out.columns)
    assert "isFraud" in out.columns


def test_time_split_is_chronological_and_disjoint(tiny_df):
    split = time_split(tiny_df, train_end=10, val_end=15)
    assert split.train["step"].max() <= 10
    assert split.val["step"].min() > 10 and split.val["step"].max() <= 15
    assert split.test["step"].min() > 15
    assert len(split.train) + len(split.val) + len(split.test) == len(tiny_df)


def test_time_split_rejects_bad_boundaries(tiny_df):
    with pytest.raises(ValueError):
        time_split(tiny_df, train_end=15, val_end=10)
    with pytest.raises(ValueError):
        time_split(tiny_df, train_end=10, val_end=20)  # empty test set


def test_summarize_split_counts_frauds(tiny_df):
    summary = summarize_split(time_split(tiny_df, train_end=10, val_end=15))
    assert summary["frauds"].sum() == 2
    assert list(summary.index) == ["train", "val", "test"]


def test_required_columns_cover_label():
    assert "isFraud" in REQUIRED_COLUMNS
