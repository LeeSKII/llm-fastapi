from fastapi import FastAPI
from .core.config import settings
from .routers import api

app = FastAPI(lifespan=settings['lifespan'])

app.include_router(api.api_router)
