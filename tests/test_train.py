"""Tests for the training pipeline."""

import pandas as pd
import pytest

from src.train import FEATURE_COLUMNS, ModelTrainer


def test_rmse_score_returns_float() -> None:
    """RMSE helper should return zero for identical predictions."""
    actual = pd.Series([1.0, 2.0, 3.0])
    predicted = [1.0, 2.0, 3.0]

    result = ModelTrainer.rmse_score(actual, predicted)

    assert isinstance(result, float)
    assert result == 0.0


def test_build_features_and_target_success(sample_dataframe: pd.DataFrame) -> None:
    """Feature/target split should return the expected columns."""
    trainer = ModelTrainer()
    x_data, y_data = trainer.build_features_and_target(sample_dataframe)

    assert list(x_data.columns) == FEATURE_COLUMNS
    assert len(y_data) == len(sample_dataframe)


def test_build_features_and_target_raises_on_missing_columns(
    sample_dataframe: pd.DataFrame,
) -> None:
    """It should fail if required columns are missing."""
    trainer = ModelTrainer()
    broken_df = sample_dataframe.drop(columns=["rm"])

    with pytest.raises(ValueError, match="Missing required columns"):
        trainer.build_features_and_target(broken_df)


def test_load_training_data_reads_csv(tmp_path, sample_dataframe: pd.DataFrame) -> None:
    """Training data loader should read existing CSV files."""
    csv_path = tmp_path / "train.csv"
    sample_dataframe.to_csv(csv_path, index=False)

    trainer = ModelTrainer()
    loaded_df = trainer.load_training_data(csv_path)

    assert not loaded_df.empty
    assert list(loaded_df.columns) == list(sample_dataframe.columns)


def test_load_training_data_raises_if_file_missing(tmp_path) -> None:
    """Training data loader should raise for a missing file."""
    trainer = ModelTrainer()

    with pytest.raises(FileNotFoundError):
        trainer.load_training_data(tmp_path / "missing.csv")


def test_run_trains_and_saves_artifacts(tmp_path, sample_dataframe: pd.DataFrame) -> None:
    """End-to-end: run() should fit a model and persist artifacts."""
    train_csv = tmp_path / "train.csv"
    sample_dataframe.to_csv(train_csv, index=False)

    trainer = ModelTrainer()
    trainer.data_path = train_csv
    trainer.model_path = tmp_path / "model.joblib"
    trainer.metrics_path = tmp_path / "metrics.json"

    _, rmse = trainer.run()

    assert isinstance(rmse, float)
    assert trainer.model_path.exists()
    assert trainer.metrics_path.exists()
