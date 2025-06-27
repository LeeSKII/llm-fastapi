from typing import Union

from fastapi import FastAPI

from .scheduler.start import lifespan
from .routers import reports
from ..utils import logger

from dotenv import load_dotenv
load_dotenv()


app = FastAPI(lifespan=lifespan)

app.include_router(reports.router,prefix="/reports")

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}