"""Train the occupancy prediction model.

Usage:
    python -m src.ml.train
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MODEL_PATH


def load_training_data() -> pd.DataFrame:
    """Load occupancy history and build the feature table."""
    # TODO: database.load_occupancy() -> features.build_features().
    raise NotImplementedError


def train_model(df: pd.DataFrame) -> Any:
    """Fit a regression model predicting free spots.

    Returns:
        A fitted scikit-learn estimator / pipeline.
    """
    # TODO: one-hot encode garage_id, e.g. RandomForestRegressor or GradientBoosting.
    # TODO: time-based train/test split (no random shuffle!).
    raise NotImplementedError


def evaluate(model: Any, df: pd.DataFrame) -> dict[str, float]:
    """Return evaluation metrics, e.g. {"mae": ..., "rmse": ...}."""
    # TODO: implement and compare against a naive baseline (same hour last week).
    raise NotImplementedError


def save_model(model: Any, path: Path = MODEL_PATH) -> None:
    """Persist the model with joblib."""
    # TODO: implement.
    raise NotImplementedError


def main() -> None:
    df = load_training_data()
    model = train_model(df)
    print(evaluate(model, df))
    save_model(model)


if __name__ == "__main__":
    main()
