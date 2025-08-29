# llm-fastapi

bunch of fastapi functions

## Development

start service: `uv run uvicorn src.app.main:app --reload`

## Deployment

### Uvicorn

`uv run uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --workers 4`

### PM2

`pm2 start ecosystem.config.js`
