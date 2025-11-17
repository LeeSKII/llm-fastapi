from contextlib import asynccontextmanager
from fastapi import FastAPI

from ..database.mysql_pool import mysql_pool
from ..scheduler import create_scheduler, start_scheduler, shutdown_scheduler

import logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    scheduler = None
    try:
        # 创建并启动调度器
        scheduler = create_scheduler()
        start_scheduler(scheduler)

        await mysql_pool.create_pool()  # 创建 MySQL 连接池
        
        logging.info("✅ 应用启动完成")
        
        yield  # FastAPI 运行期间保持
        
        # 关闭调度器
        if scheduler:
            shutdown_scheduler(scheduler)
        
    except Exception as e:
        logging.error(f"❌ 应用启动失败: {e}")
        raise
    
    finally:
        await mysql_pool.close_pool()  # 关闭 MySQL 连接池
        logging.info("✅ 应用已关闭")