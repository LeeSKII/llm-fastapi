from fastapi import APIRouter
from pydantic import BaseModel, Field
import logging

from ....utils.llm_graph import translate_graph

router = APIRouter()

class TranslateInput(BaseModel):
    input_text: str = Field(..., description="待翻译的文本")
    target_language: str = Field(default="英文", description="目标语言")

@router.get("/", tags=["translate"])
async def test():
    return {"message": "Hello, translate!"}

# LLM审查
@router.post("/run",tags=["translate"])
async def run_workflow(input_data: TranslateInput):
    logging.info(input_data)
    result = await translate_graph.graph.ainvoke({"input": input_data.input_text, "target_language": input_data.target_language})
    return result

