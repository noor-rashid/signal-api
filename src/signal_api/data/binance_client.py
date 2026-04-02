"""Binance REST + WebSocket client for crypto OHLCV data."""

import asyncio
import json
import logging
from datetime import datetime, timezone

import httpx
import pandas as pd
import websockets

logger = logging.getLogger(__name__)

BASE_URL = "https://api.binance.com"
FUTURES_URL = "https://fapi.binance.com"
WS_URL = "wss://stream.binance.com:9443/ws"

# Binance kline intervals
VALID_INTERVALS = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
}

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def _parse_klines(raw: list[list]) -> pd.DataFrame:
    """Parse raw Binance kline data into a DataFrame."""
    df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
    df = df.drop(columns=["ignore"])

    # Convert timestamps to datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Convert numeric columns
    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_volume", "taker_buy_volume", "taker_buy_quote_volume",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    df["trades"] = df["trades"].astype(int)

    return df


VALID_FUTURES_PERIODS = {
    "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d",
}


class BinanceClient:
    """Fetch historical and real-time crypto OHLCV data from Binance."""

    def __init__(self, base_url: str = BASE_URL, futures_url: str = FUTURES_URL):
        self.base_url = base_url
        self.futures_url = futures_url

    async def fetch_historical(
        self,
        symbol: str,
        interval: str = "1h",
        start: str | None = None,
        end: str | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch historical klines from Binance REST API.

        Args:
            symbol: Trading pair, e.g. "BTCUSDT"
            interval: Kline interval, e.g. "1h", "4h", "1d"
            start: Start time as ISO string, e.g. "2025-01-01"
            end: End time as ISO string
            limit: Max candles per request (Binance max: 1000)

        Returns:
            DataFrame with OHLCV data
        """
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'. Must be one of {VALID_INTERVALS}")

        params: dict[str, str | int] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }
        if start:
            params["startTime"] = int(
                datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1000
            )
        if end:
            params["endTime"] = int(
                datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp() * 1000
            )

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/api/v3/klines",
                params=params,
                timeout=30.0,
            )
            resp.raise_for_status()
            raw = resp.json()

        if not raw:
            return pd.DataFrame(columns=KLINE_COLUMNS[:-1])

        df = _parse_klines(raw)
        df["symbol"] = symbol.upper()
        logger.info(f"Fetched {len(df)} klines for {symbol.upper()} ({interval})")
        return df

    async def fetch_full_history(
        self,
        symbol: str,
        interval: str = "1h",
        start: str = "2024-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch complete history by paginating through Binance API."""
        all_frames: list[pd.DataFrame] = []
        current_start = start

        while True:
            df = await self.fetch_historical(
                symbol=symbol,
                interval=interval,
                start=current_start,
                end=end,
                limit=1000,
            )
            if df.empty:
                break

            all_frames.append(df)

            # Move start to after last candle
            last_close_time = df["close_time"].iloc[-1]
            current_start = (last_close_time + pd.Timedelta(milliseconds=1)).isoformat()

            # If we got fewer than 1000, we've reached the end
            if len(df) < 1000:
                break

            # Rate limit courtesy
            await asyncio.sleep(0.1)

        if not all_frames:
            return pd.DataFrame()

        result = pd.concat(all_frames, ignore_index=True)
        result = result.drop_duplicates(subset=["open_time", "symbol"]).reset_index(drop=True)
        logger.info(f"Full history: {len(result)} total klines for {symbol}")
        return result

    async def _paginate_futures(
        self,
        endpoint: str,
        symbol: str,
        period: str | None = None,
        limit: int = 500,
        start: str | None = None,
        max_age_days: int | None = 30,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Generic paginated fetcher for Binance Futures data endpoints."""
        params: dict[str, str | int] = {"symbol": symbol.upper(), "limit": limit}
        if period:
            if period not in VALID_FUTURES_PERIODS:
                raise ValueError(f"Invalid period '{period}'. Must be one of {VALID_FUTURES_PERIODS}")
            params["period"] = period

        # Clamp start time for endpoints with limited history
        if max_age_days:
            min_start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max_age_days)
            if start:
                start_ts = pd.Timestamp(start, tz="UTC")
                if start_ts < min_start:
                    start = min_start.isoformat()
            else:
                start = min_start.isoformat()

        if start:
            params["startTime"] = int(
                pd.Timestamp(start, tz="UTC").timestamp() * 1000
            )

        all_frames: list[pd.DataFrame] = []
        async with httpx.AsyncClient() as client:
            while True:
                resp = await client.get(
                    f"{self.futures_url}{endpoint}",
                    params=params,
                    timeout=30.0,
                )
                resp.raise_for_status()
                raw = resp.json()

                if not raw:
                    break

                df = pd.DataFrame(raw)
                all_frames.append(df)

                if len(df) < limit:
                    break

                # Advance startTime past the last record
                last_ts = int(df[timestamp_col].iloc[-1])
                params["startTime"] = last_ts + 1

                await asyncio.sleep(0.1)

        if not all_frames:
            return pd.DataFrame()

        result = pd.concat(all_frames, ignore_index=True)
        result[timestamp_col] = pd.to_datetime(
            result[timestamp_col].astype(int), unit="ms", utc=True
        )
        result["symbol"] = symbol.upper()

        # Convert numeric columns
        for col in result.columns:
            if col not in (timestamp_col, "symbol"):
                try:
                    result[col] = pd.to_numeric(result[col])
                except (ValueError, TypeError):
                    pass

        result = result.drop_duplicates(subset=[timestamp_col, "symbol"]).reset_index(drop=True)
        logger.info(f"Fetched {len(result)} records from {endpoint} for {symbol}")
        return result

    async def fetch_open_interest_history(
        self,
        symbol: str,
        period: str = "1h",
        start: str | None = None,
    ) -> pd.DataFrame:
        """Fetch Open Interest history (last 30 days only)."""
        return await self._paginate_futures(
            endpoint="/futures/data/openInterestHist",
            symbol=symbol,
            period=period,
            limit=500,
            start=start,
            max_age_days=30,
            timestamp_col="timestamp",
        )

    async def fetch_funding_rate(
        self,
        symbol: str,
        start: str | None = None,
    ) -> pd.DataFrame:
        """Fetch Funding Rate history (deep history available)."""
        df = await self._paginate_futures(
            endpoint="/fapi/v1/fundingRate",
            symbol=symbol,
            period=None,
            limit=1000,
            start=start,
            max_age_days=None,  # No limit — deep history
            timestamp_col="fundingTime",
        )
        if not df.empty:
            df = df.rename(columns={"fundingTime": "timestamp"})
        return df

    async def fetch_long_short_ratio(
        self,
        symbol: str,
        period: str = "1h",
        start: str | None = None,
    ) -> pd.DataFrame:
        """Fetch Global Long/Short Ratio (last 30 days only)."""
        return await self._paginate_futures(
            endpoint="/futures/data/globalLongShortAccountRatio",
            symbol=symbol,
            period=period,
            limit=500,
            start=start,
            max_age_days=30,
            timestamp_col="timestamp",
        )

    async def fetch_taker_buy_sell_volume(
        self,
        symbol: str,
        period: str = "1h",
        start: str | None = None,
    ) -> pd.DataFrame:
        """Fetch Taker Buy/Sell Volume ratio (last 30 days only)."""
        return await self._paginate_futures(
            endpoint="/futures/data/takerlongshortRatio",
            symbol=symbol,
            period=period,
            limit=500,
            start=start,
            max_age_days=30,
            timestamp_col="timestamp",
        )

    async def stream_klines(
        self,
        symbol: str,
        interval: str = "1m",
        callback: asyncio.coroutines = None,  # type: ignore[assignment]
    ) -> None:
        """Stream real-time klines via Binance WebSocket.

        Args:
            symbol: Trading pair, e.g. "BTCUSDT"
            interval: Kline interval
            callback: Async function called with each kline dict
        """
        stream = f"{symbol.lower()}@kline_{interval}"
        uri = f"{WS_URL}/{stream}"

        logger.info(f"Connecting to WebSocket: {stream}")

        async for ws in websockets.connect(uri):
            try:
                async for message in ws:
                    data = json.loads(message)
                    kline = data["k"]

                    parsed = {
                        "symbol": kline["s"],
                        "open_time": pd.Timestamp(kline["t"], unit="ms", tz="UTC"),
                        "open": float(kline["o"]),
                        "high": float(kline["h"]),
                        "low": float(kline["l"]),
                        "close": float(kline["c"]),
                        "volume": float(kline["v"]),
                        "quote_volume": float(kline["q"]),
                        "trades": int(kline["n"]),
                        "is_closed": kline["x"],
                    }

                    if callback:
                        await callback(parsed)

            except websockets.ConnectionClosed:
                logger.warning("WebSocket disconnected, reconnecting...")
                continue
