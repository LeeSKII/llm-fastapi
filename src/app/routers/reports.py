from fastapi import APIRouter
from ...utils.report_utils import post_report_with_llm

router = APIRouter()


@router.get("/", tags=["reports"])
async def test():
    return {"message": "Hello, reports!"}

@router.get("/post-report-weekly", tags=["reports"])
async def weekly_report():
    response = await post_report_with_llm()
    return {"message": response}
