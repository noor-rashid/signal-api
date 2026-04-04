"""Signal API — Production ML API for Financial Signal Detection."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from signal_api.data.binance_client import BinanceClient
from signal_api.data.cache import ParquetCache
from signal_api.features import build_feature_matrix
from signal_api.models.serve import SignalPredictor

logger = logging.getLogger(__name__)

# Global state
predictor: SignalPredictor | None = None
SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global predictor
    model_path = Path("models/BTCUSDT_logistic_regression.pkl")
    if model_path.exists():
        predictor = SignalPredictor()
        predictor.load(model_path)
        logger.info("Model loaded on startup")
    else:
        logger.warning(f"No model found at {model_path} — /predict will return 503")
    yield


app = FastAPI(
    title="Signal API",
    description="Production ML API for Crypto Tail Risk Detection",
    version="0.2.0",
    lifespan=lifespan,
)


# --- Request/Response Models ---

class PredictRequest(BaseModel):
    symbol: str = Field(default="BTCUSDT", description="Trading pair (e.g. BTCUSDT)")
    use_latest: bool = Field(default=True, description="Use latest cached data for prediction")


class PredictResponse(BaseModel):
    symbol: str
    tail_probability: float = Field(description="Probability of a left-tail event (0-1)")
    risk_level: str = Field(description="NORMAL / ELEVATED / HIGH / CRITICAL")
    features_used: dict[str, float]
    horizon: str = Field(default="4h")
    threshold: str = Field(default="bottom 5%")


class SignalResponse(BaseModel):
    symbol: str
    signals: dict[str, float | None]
    timestamp: str


# --- Endpoints ---

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/ready")
def ready() -> dict[str, bool]:
    model_loaded = predictor is not None and predictor.model is not None
    return {"ready": model_loaded}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict tail risk probability for a symbol.

    Returns probability of a left-tail event (worst 5% of returns)
    over the next 4 hours.
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    symbol = request.symbol.upper()
    if symbol not in SUPPORTED_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Unsupported symbol. Use: {SUPPORTED_SYMBOLS}")

    cache = ParquetCache()

    # Load latest data
    spot = cache.load(symbol, "1h")
    if spot.empty:
        raise HTTPException(status_code=404, detail=f"No cached data for {symbol}")

    funding = cache.load(symbol, "all", prefix="FUNDING_")
    oi = cache.load(symbol, "1h", prefix="OI_")
    ls = cache.load(symbol, "1h", prefix="LSRATIO_")

    # Build features on latest data
    matrix = build_feature_matrix(spot, oi, funding, ls)

    result = predictor.predict(matrix)

    return PredictResponse(
        symbol=symbol,
        tail_probability=result["tail_probability"],
        risk_level=result["risk_level"],
        features_used=result["features_used"],
    )


@app.get("/signals/{symbol}", response_model=SignalResponse)
async def get_signals(symbol: str) -> SignalResponse:
    """Get current signal values for a symbol (no prediction, just features)."""
    symbol = symbol.upper()
    if symbol not in SUPPORTED_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Unsupported symbol. Use: {SUPPORTED_SYMBOLS}")

    cache = ParquetCache()
    spot = cache.load(symbol, "1h")
    if spot.empty:
        raise HTTPException(status_code=404, detail=f"No cached data for {symbol}")

    funding = cache.load(symbol, "all", prefix="FUNDING_")
    oi = cache.load(symbol, "1h", prefix="OI_")
    ls = cache.load(symbol, "1h", prefix="LSRATIO_")

    matrix = build_feature_matrix(spot, oi, funding, ls)

    # Get latest signal values
    latest = matrix.iloc[-1]
    signal_cols = [
        "taker_buy_ratio_zscore", "funding_rate_zscore",
        "realized_vol", "vol_zscore", "vol_of_vol",
        "downside_vol_ratio", "return_skewness",
        "tail_concentration", "funding_vol_spread",
    ]

    signals = {}
    for col in signal_cols:
        if col in matrix.columns:
            val = latest[col]
            signals[col] = round(float(val), 4) if not (val != val) else None

    return SignalResponse(
        symbol=symbol,
        signals=signals,
        timestamp=str(latest.get("open_time", "")),
    )
