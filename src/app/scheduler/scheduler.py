"""
调度器管理模块
负责创建、配置和管理 APScheduler 实例
"""

from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

from .task_config import TASKS


def create_scheduler(timezone: str = "Asia/Shanghai") -> BackgroundScheduler:
    """
    创建并配置调度器实例
    
    Args:
        timezone: 时区设置，默认为 Asia/Shanghai
        
    Returns:
        BackgroundScheduler: 配置好的调度器实例
    """
    scheduler = BackgroundScheduler(timezone=ZoneInfo(timezone))
    
    # 添加所有配置的任务
    for task in TASKS:
        scheduler.add_job(
            task["func"],
            task["trigger"],
            id=task["id"],
            name=task.get("name", task["id"]),
            **task.get("kwargs", {})
        )
        logging.info(f"✅ 已添加定时任务: {task.get('name', task['id'])}")
    
    return scheduler


def start_scheduler(scheduler: BackgroundScheduler) -> None:
    """
    启动调度器
    
    Args:
        scheduler: 调度器实例
    """
    scheduler.start()
    logging.info("✅ 定时任务调度器已启动")


def shutdown_scheduler(scheduler: BackgroundScheduler) -> None:
    """
    关闭调度器
    
    Args:
        scheduler: 调度器实例
    """
    scheduler.shutdown()
    logging.info("❌ 定时任务调度器已停止")