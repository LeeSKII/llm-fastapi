from fastapi import APIRouter
from pydantic import BaseModel, Field
import logging

from ....utils.llm_graph import tech_report_graph

router = APIRouter()

class ReportInput(BaseModel):
    report_id: str = Field(..., description="报告ID")
    report_year: str = Field(..., description="报告年份")

@router.get("/", tags=["tech report"])
async def test():
    return {"message": "Hello, tech report!"}

# LLM报告生成
@router.post("/run",tags=["tech report"])
async def run_workflow(input_data: ReportInput):
    logging.info(input_data)
    result = await tech_report_graph.graph.ainvoke({"report_id": input_data.report_id, "report_year": input_data.report_year})
    return result

