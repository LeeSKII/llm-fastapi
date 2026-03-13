import logging
from typing import Optional, Dict, Any, List, Union
from .mysql_pool import mysql_pool, MySQLPool
from .dm_pool import dm_pool, DMPool

class DatabaseManager:
    """数据库管理器，统一管理MySQL和达梦数据库连接池"""
    
    def __init__(self):
        self.mysql_pool = mysql_pool
        self.dm_pool = dm_pool
    
    async def initialize_all(self) -> Dict[str, bool]:
        """
        初始化所有数据库连接池
        返回: {"mysql": True/False, "dm": True/False} 表示各数据库初始化状态
        """
        results = {
            "mysql": False,
            "dm": False
        }

        # 初始化MySQL（失败不影响达梦）
        try:
            results["mysql"] = await self.mysql_pool.create_pool()
        except Exception as e:
            logging.error(f"❌ MySQL连接池初始化异常: {e}")
            results["mysql"] = False

        # 初始化达梦（失败不影响MySQL）
        try:
            results["dm"] = await self.dm_pool.create_pool()
        except Exception as e:
            logging.error(f"❌ 达梦数据库连接池初始化异常: {e}")
            results["dm"] = False

        # 根据初始化结果输出日志
        if results["mysql"] and results["dm"]:
            logging.info("✅ 所有数据库连接池初始化完成")
        elif results["mysql"] or results["dm"]:
            available = []
            if results["mysql"]:
                available.append("MySQL")
            if results["dm"]:
                available.append("达梦")
            logging.warning(f"⚠️ 部分数据库连接失败，可用: {', '.join(available)}")
        else:
            logging.error("❌ 所有数据库连接池初始化失败，服务以降级模式启动")

        return results
    
    async def close_all(self):
        """关闭所有数据库连接池"""
        try:
            await self.mysql_pool.close_pool()
            await self.dm_pool.close_pool()
            logging.info("✅ 所有数据库连接池已关闭")
        except Exception as e:
            logging.error(f"❌ 关闭数据库连接池失败: {e}")
    
    def get_mysql_pool(self) -> MySQLPool:
        """获取MySQL连接池"""
        return self.mysql_pool
    
    def get_dm_pool(self) -> DMPool:
        """获取达梦数据库连接池"""
        return self.dm_pool
    
    async def health_check(self) -> Dict[str, bool]:
        """检查所有数据库连接状态"""
        status = {
            "mysql": False,
            "dm": False
        }
        
        # 检查MySQL连接
        try:
            await self.mysql_pool.fetch_one("SELECT 1")
            status["mysql"] = True
        except Exception as e:
            logging.error(f"MySQL健康检查失败: {e}")
        
        # 检查达梦数据库连接
        try:
            await self.dm_pool.fetch_one("SELECT 1")
            status["dm"] = True
        except Exception as e:
            logging.error(f"达梦数据库健康检查失败: {e}")
        
        return status
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """获取所有连接池状态"""
        return {
            "mysql": {
                "initialized": self.mysql_pool.is_initialized(),
                "error": self.mysql_pool.get_init_error(),
                "pool_size": self.mysql_pool.pool.size if self.mysql_pool.pool else 0,
                "pool_free": self.mysql_pool.pool.freesize if self.mysql_pool.pool else 0,
            },
            "dm": {
                "initialized": self.dm_pool.is_initialized(),
                "error": self.dm_pool.get_init_error(),
                "pool_size": self.dm_pool.pool.qsize() if self.dm_pool.pool else 0,
                "semaphore_available": self.dm_pool.semaphore._value if self.dm_pool.semaphore else 0,
            }
        }

# 全局数据库管理器实例
db_manager = DatabaseManager()