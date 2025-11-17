from typing import Union

from fastapi import APIRouter

from .llm import doc, chat, translate, contract, tech_report, contract_checker
from .common import yus

api_router = APIRouter()

api_router.include_router(doc.router, prefix="/llm/doc")
api_router.include_router(chat.router, prefix="/llm/chat")
api_router.include_router(translate.router, prefix="/llm/translate")
api_router.include_router(contract.router, prefix="/llm/contract")
api_router.include_router(tech_report.router, prefix="/llm/tech_report")
api_router.include_router(contract_checker.router, prefix="/llm/contract_checker")

api_router.include_router(yus.router, prefix="/yus")

@api_router.get("/health")
def health_check():
    return {"status": "heathy"}


@api_router.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}