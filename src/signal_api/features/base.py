"""Base class for all signal features."""

from abc import ABC, abstractmethod

import pandas as pd


class Feature(ABC):
    """Abstract base for a single signal feature."""

    name: str
    required_columns: list[str]
    lookback_periods: int

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute the feature and return a Series aligned to df index."""

    def validate_input(self, df: pd.DataFrame) -> None:
        """Check that required columns exist."""
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.name} requires columns {missing} which are not in DataFrame"
            )
