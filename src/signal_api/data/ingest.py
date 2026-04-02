"""High-level ingestion: backfill + incremental updates."""

import asyncio
import logging
from pathlib import Path

import pandas as pd

from signal_api.data.binance_client import BinanceClient
from signal_api.data.cache import ParquetCache

logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_INTERVAL = "1h"
DEFAULT_START = "2024-01-01"


async def backfill(
    symbols: list[str] | None = None,
    interval: str = DEFAULT_INTERVAL,
    start: str = DEFAULT_START,
    data_dir: Path = Path("data/raw"),
) -> dict[str, int]:
    """Backfill historical data for given symbols.

    Returns dict of symbol -> row count.
    """
    symbols = symbols or DEFAULT_SYMBOLS
    client = BinanceClient()
    cache = ParquetCache(data_dir)
    results: dict[str, int] = {}

    for symbol in symbols:
        # Resume from where we left off
        latest = cache.latest_timestamp(symbol, interval)
        effective_start = start
        if latest is not None:
            effective_start = (latest + pd.Timedelta(milliseconds=1)).isoformat()
            logger.info(f"{symbol}: resuming from {effective_start}")

        df = await client.fetch_full_history(
            symbol=symbol,
            interval=interval,
            start=effective_start,
        )

        if df.empty:
            logger.info(f"{symbol}: already up to date")
            results[symbol] = 0
            continue

        cache.save(df, symbol, interval)
        results[symbol] = len(df)

    return results


async def validate_data(
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
    data_dir: Path = Path("data/raw"),
) -> dict[str, bool | int | str]:
    """Run validation checks on cached data."""
    cache = ParquetCache(data_dir)
    df = cache.load(symbol, interval)

    if df.empty:
        return {"valid": False, "error": "No data found"}

    checks: dict[str, bool | int | str] = {
        "valid": True,
        "rows": len(df),
        "date_range": f"{df['open_time'].min()} to {df['open_time'].max()}",
        "has_nulls": bool(df[["open", "high", "low", "close", "volume"]].isnull().any().any()),
        "has_duplicates": bool(df.duplicated(subset=["open_time", "symbol"]).any()),
        "ohlc_valid": bool((df["high"] >= df["low"]).all()),
        "volume_positive": bool((df["volume"] >= 0).all()),
    }

    if checks["has_nulls"] or checks["has_duplicates"] or not checks["ohlc_valid"]:
        checks["valid"] = False

    return checks


FUTURES_DATA_TYPES = {
    "OI": ("fetch_open_interest_history", "1h"),
    "FUNDING": ("fetch_funding_rate", None),
    "LSRATIO": ("fetch_long_short_ratio", "1h"),
    "TAKER": ("fetch_taker_buy_sell_volume", "1h"),
}


async def backfill_futures(
    symbols: list[str] | None = None,
    data_dir: Path = Path("data/raw"),
) -> dict[str, dict[str, int]]:
    """Backfill all futures data types for given symbols.

    Returns nested dict: symbol -> data_type -> row count.
    """
    symbols = symbols or DEFAULT_SYMBOLS
    client = BinanceClient()
    cache = ParquetCache(data_dir)
    results: dict[str, dict[str, int]] = {}

    for symbol in symbols:
        results[symbol] = {}
        for data_type, (method_name, period) in FUTURES_DATA_TYPES.items():
            method = getattr(client, method_name)
            interval_key = period or "all"
            prefix = f"{data_type}_"

            # For funding rate (deep history), resume from latest
            start = None
            if data_type == "FUNDING":
                latest = cache.latest_timestamp(
                    symbol, interval_key, prefix=prefix, timestamp_col="timestamp"
                )
                if latest is not None:
                    start = (latest + pd.Timedelta(milliseconds=1)).isoformat()
                    logger.info(f"{symbol}/{data_type}: resuming from {start}")
                else:
                    start = DEFAULT_START

            kwargs: dict = {"symbol": symbol}
            if period:
                kwargs["period"] = period
            if start:
                kwargs["start"] = start

            try:
                df = await method(**kwargs)
                if df.empty:
                    logger.info(f"{symbol}/{data_type}: no new data")
                    results[symbol][data_type] = 0
                    continue

                cache.save(
                    df, symbol, interval_key,
                    prefix=prefix, timestamp_col="timestamp",
                )
                results[symbol][data_type] = len(df)
            except Exception as e:
                logger.error(f"{symbol}/{data_type}: {e}")
                results[symbol][data_type] = 0

    return results


def run_backfill(
    symbols: list[str] | None = None,
    interval: str = DEFAULT_INTERVAL,
    start: str = DEFAULT_START,
) -> dict[str, int]:
    """Sync wrapper for backfill."""
    return asyncio.run(backfill(symbols=symbols, interval=interval, start=start))


def run_backfill_futures(
    symbols: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Sync wrapper for futures backfill."""
    return asyncio.run(backfill_futures(symbols=symbols))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Spot Data ===")
    results = run_backfill()
    for sym, count in results.items():
        print(f"  {sym}: {count} rows ingested")

    print("\n=== Futures Data ===")
    futures_results = run_backfill_futures()
    for sym, types in futures_results.items():
        for dtype, count in types.items():
            print(f"  {sym}/{dtype}: {count} rows")
