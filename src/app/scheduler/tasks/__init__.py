"""
Tasks 包初始化文件
统一导出所有任务函数
"""

from .reports import send_weekly_report

__all__ = [
    "send_weekly_report"
]