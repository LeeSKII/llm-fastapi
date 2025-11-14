from ...utils import logger
from ..scheduler.start import lifespan
from typing import TypedDict
import os

class Settings(TypedDict):
    app_name: str
    lifespan: object

def get_settings() -> Settings:
    return {
        "app_name": os.getenv("APP_NAME", "LLM-FastAPI"),
        "lifespan": lifespan
    }

settings = get_settings()