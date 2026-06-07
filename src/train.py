"""Training pipeline for the Boston Housing regression model.

The pipeline is organised as a ModelTrainer class that encapsulates the whole
lifecycle: loading configuration and data, building features, creating and
fitting the estimator, evaluating it and persisting the artifacts.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from configparser import ConfigParser
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.config import BASE_DIR, load_config

FEATURE_COLUMNS = [
    "crim",
    "zn",
    "indus",
    "chas",
    "nox",
    "rm",
    "age",
    "dis",
    "rad",
    "tax",
    "ptratio",
    "black",
    "lstat",
]


class ModelTrainer:
    """Encapsulates the Boston Housing model training pipeline."""

    feature_columns = FEATURE_COLUMNS

    def __init__(self, config: Optional[ConfigParser] = None) -> None:
        """Initialise the trainer from project configuration.

        Args:
            config: Optional pre-loaded configuration. Loaded from config.ini
                when omitted.
        """
        self.config = config or load_config()
        self.data_path = BASE_DIR / self.config["paths"]["train_data"]
        self.model_path = BASE_DIR / self.config["paths"]["model_path"]
        self.metrics_path = BASE_DIR / "metrics.json"

        self.random_state = self.config.getint("project", "random_state")
        self.target_col = self.config["project"]["target_col"]
        self.id_col = self.config["project"]["id_col"]
        self.test_size = self.config.getfloat("training", "test_size")

    @staticmethod
    def rmse_score(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        mse = mean_squared_error(y_true, y_pred)
        return float(np.sqrt(mse))

    def load_training_data(self, data_path: Optional[Path] = None) -> pd.DataFrame:
        """Load the training dataset from CSV.

        Raises:
            FileNotFoundError: If the training file does not exist.
        """
        resolved = data_path or self.data_path
        if not resolved.exists():
            raise FileNotFoundError(f"Training file not found: {resolved}")
        return pd.read_csv(resolved)

    def build_features_and_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Split a dataframe into the feature matrix and target vector.

        Raises:
            ValueError: If required columns are missing.
        """
        required_columns = set(self.feature_columns + [self.target_col, self.id_col])
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in train dataset: {sorted(missing)}"
            )

        x_data = df[self.feature_columns].copy()
        y_data = df[self.target_col].copy()
        return x_data, y_data

    def create_model(self) -> RandomForestRegressor:
        """Create a RandomForestRegressor configured from config.ini."""
        max_depth_raw = self.config["model"].get("max_depth", "").strip()
        max_depth = int(max_depth_raw) if max_depth_raw else None

        return RandomForestRegressor(
            n_estimators=self.config.getint("model", "n_estimators"),
            max_depth=max_depth,
            min_samples_split=self.config.getint("model", "min_samples_split"),
            min_samples_leaf=self.config.getint("model", "min_samples_leaf"),
            random_state=self.random_state,
            n_jobs=self.config.getint("model", "n_jobs"),
        )

    def save_model(
        self, model: RandomForestRegressor, model_path: Optional[Path] = None
    ) -> None:
        """Persist the trained model to disk."""
        resolved = model_path or self.model_path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, resolved)

    def save_metrics(self, rmse: float, metrics_path: Optional[Path] = None) -> None:
        """Save training metrics to a JSON file."""
        resolved = metrics_path or self.metrics_path
        metrics = {
            "rmse": round(rmse, 6),
            "model_name": "RandomForestRegressor",
            "random_state": self.random_state,
            "test_size": self.test_size,
            "n_estimators": self.config.getint("model", "n_estimators"),
        }
        resolved.write_text(
            json.dumps(metrics, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )

    def run(self) -> Tuple[RandomForestRegressor, float]:
        """Train and evaluate the model, then persist artifacts.

        Returns:
            Tuple of the trained model and the validation RMSE.
        """
        df = self.load_training_data()
        x_data, y_data = self.build_features_and_target(df)

        x_train, x_valid, y_train, y_valid = train_test_split(
            x_data,
            y_data,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        model = self.create_model()
        model.fit(x_train, y_train)

        rmse = self.rmse_score(y_valid, model.predict(x_valid))

        self.save_model(model)
        self.save_metrics(rmse)

        print(f"Model saved to: {self.model_path}")
        print(f"Metrics saved to: {self.metrics_path}")
        print(f"Validation RMSE: {rmse:.4f}")

        return model, rmse


def main() -> None:
    """Entry point used by the DVC pipeline (`python -m src.train`)."""
    ModelTrainer().run()


if __name__ == "__main__":
    main()
