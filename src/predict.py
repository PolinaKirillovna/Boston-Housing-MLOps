"""Batch inference for Boston Housing data.

Inference is organised as a BatchPredictor class that loads the trained model,
validates incoming features and produces predictions / a submission file.
"""

from configparser import ConfigParser
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from src.config import BASE_DIR, load_config
from src.train import FEATURE_COLUMNS


class BatchPredictor:
    """Encapsulates batch prediction over Boston Housing data."""

    feature_columns = FEATURE_COLUMNS

    def __init__(self, config: Optional[ConfigParser] = None) -> None:
        """Initialise the predictor from project configuration."""
        self.config = config or load_config()
        self.model_path = BASE_DIR / self.config["paths"]["model_path"]
        self.test_data_path = BASE_DIR / self.config["paths"]["test_data"]
        self.submission_path = BASE_DIR / self.config["paths"]["submission_path"]
        self.id_col = self.config["project"]["id_col"]
        self._model = None

    def load_model(self, model_path: Optional[Path] = None):
        """Load and cache the serialized model artifact.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        resolved = model_path or self.model_path
        if not resolved.exists():
            raise FileNotFoundError(
                f"Model file not found: {resolved}. Run training first."
            )
        self._model = joblib.load(resolved)
        return self._model

    def validate_feature_columns(self, df: pd.DataFrame) -> None:
        """Validate that all required feature columns are present.

        Raises:
            ValueError: If any expected feature is missing.
        """
        missing = set(self.feature_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")

    def predict_from_dataframe(self, df: pd.DataFrame):
        """Generate predictions for a dataframe of features."""
        self.validate_feature_columns(df)
        model = self._model or self.load_model()
        return model.predict(df[self.feature_columns])

    def predict_test_file(self, test_path: Optional[Path] = None) -> pd.DataFrame:
        """Run batch prediction for a test CSV and write a submission file.

        Raises:
            FileNotFoundError: If the test file does not exist.
            ValueError: If the ID column is missing.
        """
        resolved = test_path or self.test_data_path
        if not resolved.exists():
            raise FileNotFoundError(f"Test file not found: {resolved}")

        test_df = pd.read_csv(resolved)
        if self.id_col not in test_df.columns:
            raise ValueError(f"Missing ID column: {self.id_col}")

        predictions = self.predict_from_dataframe(test_df)
        submission = pd.DataFrame(
            {self.id_col: test_df[self.id_col], "medv": predictions}
        )
        submission.to_csv(self.submission_path, index=False)

        print(f"Submission file saved to: {self.submission_path}")
        print(submission.head())
        return submission

    def run(self) -> pd.DataFrame:
        """Entry point: produce the submission from the configured test file."""
        return self.predict_test_file()


def main() -> None:
    """Entry point for `python -m src.predict`."""
    BatchPredictor().run()


if __name__ == "__main__":
    main()
