"""Tests for the batch prediction module."""

import pandas as pd
import pytest

from src.predict import BatchPredictor
from src.train import FEATURE_COLUMNS


def test_validate_feature_columns_success(sample_dataframe: pd.DataFrame) -> None:
    """Validation should pass when all features are present."""
    predictor = BatchPredictor()
    predictor.validate_feature_columns(sample_dataframe[FEATURE_COLUMNS].copy())


def test_validate_feature_columns_raises(sample_dataframe: pd.DataFrame) -> None:
    """Validation should fail when a feature is missing."""
    predictor = BatchPredictor()
    broken_df = sample_dataframe.drop(columns=["rm"])

    with pytest.raises(ValueError, match="Missing feature columns"):
        predictor.validate_feature_columns(broken_df)


def test_load_model_raises_for_missing_model(tmp_path) -> None:
    """Model loader should fail for an absent model file."""
    predictor = BatchPredictor()

    with pytest.raises(FileNotFoundError):
        predictor.load_model(tmp_path / "missing_model.joblib")


def test_predict_from_dataframe_returns_predictions(
    sample_dataframe: pd.DataFrame, trained_model
) -> None:
    """Prediction should return one value per input row."""
    predictor = BatchPredictor()
    predictor._model = trained_model
    df = sample_dataframe[FEATURE_COLUMNS].copy()

    predictions = predictor.predict_from_dataframe(df)

    assert len(predictions) == len(df)


def test_predict_test_file_creates_submission(
    tmp_path, sample_dataframe: pd.DataFrame, trained_model
) -> None:
    """Batch prediction should create a submission file."""
    test_df = sample_dataframe.drop(columns=["medv"]).copy()
    test_csv_path = tmp_path / "test.csv"
    test_df.to_csv(test_csv_path, index=False)

    predictor = BatchPredictor()
    predictor._model = trained_model
    predictor.submission_path = tmp_path / "submission.csv"

    submission = predictor.predict_test_file(test_csv_path)

    assert list(submission.columns) == ["ID", "medv"]
    assert len(submission) == len(test_df)
    assert (tmp_path / "submission.csv").exists()
