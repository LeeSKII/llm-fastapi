from contextlib import asynccontextmanager
from fastapi import FastAPI
from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ..database.mysql_pool import mysql_pool

import logging

from ..scheduler.tasks.reports import send_weekly_report

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    try:
        # 启动 APScheduler
        scheduler = BackgroundScheduler(timezone=ZoneInfo("Asia/Shanghai"))
        scheduler.add_job(
            send_weekly_report,
            # "interval", seconds=5
            CronTrigger.from_crontab("0 9 * * 1")  # 每周一 9:00
        )
        scheduler.start()

        await mysql_pool.create_pool()  # 创建 MySQL 连接池
        
        logging.info("✅ 定时任务已启动")
        
        yield  # FastAPI 运行期间保持
        
        # 关闭 APScheduler
        scheduler.shutdown()
        logging.info("❌ 定时任务已停止")
    
    except Exception as e:
        logging.error(f"❌ 应用启动失败: {e}")
        raise
    
    finally:
        await mysql_pool.close_pool()  # 关闭 MySQL 连接池
        logging.info("✅ 应用已关闭")