from typing import Union

from fastapi import FastAPI

from .scheduler.start import lifespan
from .routers import reports
from .routers.llm import doc
from ..utils import logger

app = FastAPI(lifespan=lifespan)

app.include_router(reports.router,prefix="/reports")
app.include_router(doc.router,prefix="/llm/doc")

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
