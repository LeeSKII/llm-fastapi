from fastapi import APIRouter
from pydantic import BaseModel, Field
import logging

from ....utils.llm_graph import contact_checker_graph

router = APIRouter()

class ContractCheckerInput(BaseModel):
    content: str = Field(..., description="合同正文内容")

@router.get("/", tags=["contract checker"])
async def test():
    return {"message": "Hello, contract checker!"}

# 合同检查
@router.post("/run",tags=["contract checker"])
async def run_workflow(input_data: ContractCheckerInput):
    logging.info(input_data)
    result = await contact_checker_graph.graph.ainvoke({"contract_content": input_data.content})
    return result

