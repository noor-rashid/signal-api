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

    def _path(self, symbol: str, interval: str, prefix: str = "") -> Path:
        if prefix:
            subdir = self.data_dir / "futures"
            subdir.mkdir(parents=True, exist_ok=True)
            return subdir / f"{prefix}{symbol.upper()}_{interval}.parquet"
        return self.data_dir / f"{symbol.upper()}_{interval}.parquet"

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        prefix: str = "",
        timestamp_col: str = "open_time",
    ) -> Path:
        """Save DataFrame to Parquet, merging with existing data."""
        path = self._path(symbol, interval, prefix)

        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
            dedup_cols = [timestamp_col]
            if "symbol" in df.columns:
                dedup_cols.append("symbol")
            df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
            df = df.sort_values(timestamp_col).reset_index(drop=True)

        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} rows to {path}")
        return path

    def load(self, symbol: str, interval: str, prefix: str = "") -> pd.DataFrame:
        """Load cached data from Parquet."""
        path = self._path(symbol, interval, prefix)
        if not path.exists():
            logger.warning(f"No cached data at {path}")
            return pd.DataFrame()
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df

    def exists(self, symbol: str, interval: str, prefix: str = "") -> bool:
        return self._path(symbol, interval, prefix).exists()

    def latest_timestamp(
        self,
        symbol: str,
        interval: str,
        prefix: str = "",
        timestamp_col: str = "open_time",
    ) -> pd.Timestamp | None:
        """Get the most recent timestamp in cache."""
        df = self.load(symbol, interval, prefix)
        if df.empty or timestamp_col not in df.columns:
            return None
        return df[timestamp_col].max()
