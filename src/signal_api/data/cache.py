"""Parquet-based cache for OHLCV data."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/raw")


class ParquetCache:
    """Read and write OHLCV data as partitioned Parquet files."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, interval: str) -> Path:
        return self.data_dir / f"{symbol.upper()}_{interval}.parquet"

    def save(self, df: pd.DataFrame, symbol: str, interval: str) -> Path:
        """Save DataFrame to Parquet, merging with existing data."""
        path = self._path(symbol, interval)

        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["open_time", "symbol"]).reset_index(drop=True)
            df = df.sort_values("open_time").reset_index(drop=True)

        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} rows to {path}")
        return path

    def load(self, symbol: str, interval: str) -> pd.DataFrame:
        """Load cached data from Parquet."""
        path = self._path(symbol, interval)
        if not path.exists():
            logger.warning(f"No cached data at {path}")
            return pd.DataFrame()
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df

    def exists(self, symbol: str, interval: str) -> bool:
        return self._path(symbol, interval).exists()

    def latest_timestamp(self, symbol: str, interval: str) -> pd.Timestamp | None:
        """Get the most recent open_time in cache."""
        df = self.load(symbol, interval)
        if df.empty:
            return None
        return df["open_time"].max()
