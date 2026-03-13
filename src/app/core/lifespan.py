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

        # 初始化数据库连接池（失败不阻断启动）
        db_status = await db_manager.initialize_all()

        # 将数据库状态存储到 app.state，供其他地方使用
        app.state.db_status = db_status

        # 根据初始化结果输出日志
        available_dbs = [name for name, status in db_status.items() if status]
        if available_dbs:
            logging.info(f"✅ 应用启动完成，可用数据库: {', '.join(available_dbs)}")
        else:
            logging.warning("⚠️ 应用启动完成，但无可用数据库，服务处于降级模式")

        yield  # FastAPI 运行期间保持

        # 关闭调度器
        if scheduler:
            shutdown_scheduler(scheduler)

    except Exception as e:
        # 只有非数据库相关的严重错误才阻止启动
        logging.error(f"❌ 应用启动失败: {e}")
        raise

    finally:
        await db_manager.close_all()  # 关闭所有数据库连接池
        logging.info("✅ 应用已关闭")