from fastapi import FastAPI

app = FastAPI(
    title="Signal API",
    description="Production ML API for Financial Signal Detection",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/ready")
def ready() -> dict[str, bool]:
    return {"ready": True}
