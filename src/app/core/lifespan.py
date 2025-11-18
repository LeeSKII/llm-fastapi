from contextlib import asynccontextmanager
from fastapi import FastAPI

from ..database.db_manager import db_manager
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

        await db_manager.initialize_all()  # 初始化所有数据库连接池
        
        logging.info("✅ 应用启动完成")
        
        yield  # FastAPI 运行期间保持
        
        # 关闭调度器
        if scheduler:
            shutdown_scheduler(scheduler)
        
    except Exception as e:
        logging.error(f"❌ 应用启动失败: {e}")
        raise
    
    finally:
        await db_manager.close_all()  # 关闭所有数据库连接池
        logging.info("✅ 应用已关闭")