from contextlib import asynccontextmanager
from fastapi import FastAPI
from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

from .tasks.reports import send_weekly_report

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动 APScheduler
    scheduler = BackgroundScheduler(timezone=ZoneInfo("Asia/Shanghai"))
    scheduler.add_job(
        send_weekly_report,
        # "interval", seconds=5
        CronTrigger.from_crontab("0 9 * * 1")  # 每周一 9:00
    )
    scheduler.start()
    
    logging.info("✅ 定时任务已启动")
    
    yield  # FastAPI 运行期间保持
    
    # 关闭 APScheduler
    scheduler.shutdown()
    logging.info("❌ 定时任务已停止")