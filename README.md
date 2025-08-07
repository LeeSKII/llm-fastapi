# llm-fastapi

bunch of fastapi functions

## Development

start service: `uv run uvicorn src.app.main:app --reload`

## Deployment

`uv run uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --workers 4`
