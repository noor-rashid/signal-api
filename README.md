# Signal API

Production ML API for Financial Signal Detection.

## Stack

- **FastAPI** — API framework
- **MLflow** — Experiment tracking & model registry
- **Prometheus** — Metrics & monitoring
- **Docker** — Containerisation

## Quick Start

```bash
# Local development
pip install -e ".[dev]"
uvicorn signal_api.main:app --reload

# Docker
docker compose up --build

# Run tests
pytest
```

## Services

| Service    | URL                    |
|------------|------------------------|
| API        | http://localhost:8000  |
| API Docs   | http://localhost:8000/docs |
| MLflow     | http://localhost:5000  |
| Prometheus | http://localhost:9090  |

## Project Structure

```
├── src/signal_api/    # Application code
├── tests/             # Test suite
├── notebooks/         # EDA & experiment notebooks
├── data/              # Raw and processed data
├── models/            # Saved model artifacts
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## License

MIT
