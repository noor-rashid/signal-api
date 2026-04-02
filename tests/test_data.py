"""Tests for data ingestion pipeline."""

import pandas as pd
import pytest

from signal_api.data.binance_client import BinanceClient, _parse_klines
from signal_api.data.cache import ParquetCache
from signal_api.data.ingest import validate_data


class TestParseKlines:
    def test_parse_valid_klines(self):
        raw = [[
            1704067200000, "42000.00", "42500.00", "41800.00", "42200.00",
            "100.5", 1704070800000, "4230000.00", 500,
            "60.3", "2538000.00", "0",
        ]]
        df = _parse_klines(raw)

        assert len(df) == 1
        assert df["open"].iloc[0] == 42000.00
        assert df["high"].iloc[0] == 42500.00
        assert df["low"].iloc[0] == 41800.00
        assert df["close"].iloc[0] == 42200.00
        assert df["volume"].iloc[0] == 100.5
        assert df["trades"].iloc[0] == 500
        assert "ignore" not in df.columns

    def test_parse_multiple_klines(self):
        raw = [
            [1704067200000, "42000", "42500", "41800", "42200",
             "100.5", 1704070800000, "4230000", 500, "60.3", "2538000", "0"],
            [1704070800000, "42200", "42800", "42100", "42700",
             "120.3", 1704074400000, "5120000", 600, "70.1", "2990000", "0"],
        ]
        df = _parse_klines(raw)
        assert len(df) == 2
        assert df["open_time"].is_monotonic_increasing


class TestParquetCache:
    def test_save_and_load(self, tmp_path):
        cache = ParquetCache(data_dir=tmp_path)
        df = pd.DataFrame({
            "open_time": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "open": [42000.0, 42500.0],
            "high": [42500.0, 43000.0],
            "low": [41800.0, 42200.0],
            "close": [42200.0, 42800.0],
            "volume": [100.0, 120.0],
            "symbol": ["BTCUSDT", "BTCUSDT"],
        })

        cache.save(df, "BTCUSDT", "1h")
        assert cache.exists("BTCUSDT", "1h")

        loaded = cache.load("BTCUSDT", "1h")
        assert len(loaded) == 2
        assert loaded["close"].iloc[1] == 42800.0

    def test_merge_deduplicates(self, tmp_path):
        cache = ParquetCache(data_dir=tmp_path)
        df1 = pd.DataFrame({
            "open_time": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "open": [42000.0, 42500.0],
            "symbol": ["BTCUSDT", "BTCUSDT"],
        })
        df2 = pd.DataFrame({
            "open_time": pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True),
            "open": [42500.0, 43000.0],
            "symbol": ["BTCUSDT", "BTCUSDT"],
        })

        cache.save(df1, "BTCUSDT", "1h")
        cache.save(df2, "BTCUSDT", "1h")

        loaded = cache.load("BTCUSDT", "1h")
        assert len(loaded) == 3  # No duplicates

    def test_load_nonexistent(self, tmp_path):
        cache = ParquetCache(data_dir=tmp_path)
        df = cache.load("NOPE", "1h")
        assert df.empty

    def test_latest_timestamp(self, tmp_path):
        cache = ParquetCache(data_dir=tmp_path)
        df = pd.DataFrame({
            "open_time": pd.to_datetime(["2024-01-01", "2024-06-15"], utc=True),
            "symbol": ["BTCUSDT", "BTCUSDT"],
        })
        cache.save(df, "BTCUSDT", "1h")

        latest = cache.latest_timestamp("BTCUSDT", "1h")
        assert latest == pd.Timestamp("2024-06-15", tz="UTC")


class TestBinanceClient:
    @pytest.mark.asyncio
    async def test_fetch_historical_live(self):
        """Integration test — hits real Binance API."""
        client = BinanceClient()
        df = await client.fetch_historical("BTCUSDT", interval="1d", limit=5)

        assert len(df) == 5
        assert "open" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert (df["high"] >= df["low"]).all()
        assert df["symbol"].iloc[0] == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_invalid_interval(self):
        client = BinanceClient()
        with pytest.raises(ValueError, match="Invalid interval"):
            await client.fetch_historical("BTCUSDT", interval="7m")


class TestValidation:
    @pytest.mark.asyncio
    async def test_validate_empty(self, tmp_path):
        result = await validate_data("BTCUSDT", data_dir=tmp_path)
        assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_validate_good_data(self, tmp_path):
        cache = ParquetCache(data_dir=tmp_path)
        df = pd.DataFrame({
            "open_time": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "open": [42000.0, 42500.0],
            "high": [42500.0, 43000.0],
            "low": [41800.0, 42200.0],
            "close": [42200.0, 42800.0],
            "volume": [100.0, 120.0],
            "symbol": ["BTCUSDT", "BTCUSDT"],
        })
        cache.save(df, "BTCUSDT", "1h")

        result = await validate_data("BTCUSDT", data_dir=tmp_path)
        assert result["valid"] is True
        assert result["rows"] == 2
        assert result["has_nulls"] is False
        assert result["has_duplicates"] is False
