"""
任务配置模块
集中管理所有定时任务的配置
"""

from apscheduler.triggers.cron import CronTrigger
from .tasks import send_weekly_report


# 任务配置列表
TASKS = [
    {
        "id": "weekly_report",
        "name": "每周报告发送",
        "func": send_weekly_report,
        "trigger": CronTrigger.from_crontab("0 9 * * 1"),  # 每周一 9:00
        "kwargs": {}
    },
    # 预留位置，方便后续添加更多任务
    # {
    #     "id": "daily_cleanup",
    #     "name": "每日清理任务",
    #     "func": daily_cleanup,
    #     "trigger": CronTrigger.from_crontab("0 2 * * *"),  # 每天凌晨 2:00
    #     "kwargs": {"max_age_days": 7}
    # },
    # {
    #     "id": "hourly_stats",
    #     "name": "每小时统计",
    #     "func": generate_hourly_stats,
    #     "trigger": CronTrigger.from_crontab("0 * * * *"),  # 每小时整点
    #     "kwargs": {}
    # }
]


def get_task_config(task_id: str) -> dict:
    """
    根据任务ID获取任务配置
    
    Args:
        task_id: 任务ID
        
    Returns:
        dict: 任务配置字典，如果未找到则返回 None
    """
    for task in TASKS:
        if task["id"] == task_id:
            return task
    return None


def get_all_tasks() -> list:
    """
    获取所有任务配置
    
    Returns:
        list: 任务配置列表
    """
    return TASKS