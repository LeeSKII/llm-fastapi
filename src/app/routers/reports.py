from fastapi import APIRouter
from fastapi_utils.tasks import repeat_every
from ...utils.report_utils import post_report_with_llm

router = APIRouter()


@router.get("/", tags=["reports"])
async def test():
    return {"message": "Hello, reports!"}

@router.get("/post-report-weekly", tags=["reports"])
async def weekly_report():
    response = await post_report_with_llm()
    return {"message": response}


@repeat_every(cron="0 9 * * 1")  # 每周一9:00 AM
async def weekly_task():
    print("每周一早上9点执行的任务")