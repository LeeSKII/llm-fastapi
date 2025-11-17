from fastapi import FastAPI
from .routers import api
from .core.lifespan import lifespan

app = FastAPI(lifespan=lifespan)

app.include_router(api.api_router)
