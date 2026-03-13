import aiomysql
from typing import Optional, AsyncGenerator
from ..core.config import settings
import logging

class MySQLPool:
    """MySQL连接池管理类"""

    def __init__(self):
        self.pool: Optional[aiomysql.Pool] = None
        self._initialized = False
        self._init_error: Optional[str] = None

    async def create_pool(self) -> bool:
        """
        创建MySQL连接池
        返回: True表示成功，False表示失败
        """
        try:
            self.pool = await aiomysql.create_pool(
                host=settings["MYSQL_HOST"],
                port=settings["MYSQL_PORT"],
                user=settings["MYSQL_USER"],
                password=settings["MYSQL_PASSWORD"],
                db=settings["MYSQL_DB"],
                charset=settings["MYSQL_CHARSET"],
                autocommit=True,
                minsize=settings["MYSQL_POOL_MINSIZE"],
                maxsize=settings["MYSQL_POOL_MAXSIZE"],
                pool_recycle=settings["MYSQL_POOL_RECYCLE"],
            )
            self._initialized = True
            self._init_error = None
            logging.info("✅ MySQL连接池创建成功")
            logging.info(f"📊 MySQL连接池配置: min={settings['MYSQL_POOL_MINSIZE']}, max={settings['MYSQL_POOL_MAXSIZE']}")
            return True
        except Exception as e:
            self._initialized = False
            self._init_error = str(e)
            logging.warning(f"⚠️ 创建MySQL连接池失败，服务将以降级模式运行: {e}")
            return False

    def is_initialized(self) -> bool:
        """检查连接池是否已初始化"""
        return self._initialized

    def get_init_error(self) -> Optional[str]:
        """获取初始化错误信息"""
        return self._init_error
    
    async def close_pool(self):
        """关闭连接池"""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            logging.info("✅ MySQL连接池已关闭")
    
    async def get_connection(self) -> aiomysql.Connection:
        """从连接池获取连接"""
        if not self._initialized or not self.pool:
            error_msg = "MySQL连接池未初始化"
            if self._init_error:
                error_msg += f" (错误: {self._init_error})"
            raise RuntimeError(error_msg)
        return await self.pool.acquire()
    
    async def release_connection(self, connection: aiomysql.Connection):
        """释放连接回连接池"""
        if self.pool and connection:
            await self.pool.release(connection)
    
    async def execute(self, query: str, *args) -> int:
        """执行SQL语句，返回影响的行数"""
        if not self._initialized or not self.pool:
            error_msg = "MySQL连接池未初始化"
            if self._init_error:
                error_msg += f" (错误: {self._init_error})"
            raise RuntimeError(error_msg)
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, args)
                return cur.rowcount
    
    async def fetch_one(self, query: str, *args) -> Optional[dict]:
        """查询单条记录"""
        if not self._initialized or not self.pool:
            error_msg = "MySQL连接池未初始化"
            if self._init_error:
                error_msg += f" (错误: {self._init_error})"
            raise RuntimeError(error_msg)
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, args)
                result = await cur.fetchone()
                return dict(result) if result else None
    
    async def fetch_all(self, query: str, *args) -> list:
        """查询多条记录"""
        if not self._initialized or not self.pool:
            error_msg = "MySQL连接池未初始化"
            if self._init_error:
                error_msg += f" (错误: {self._init_error})"
            raise RuntimeError(error_msg)
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, args)
                results = await cur.fetchall()
                return [dict(row) for row in results] if results else []

# 全局数据库实例
mysql_pool = MySQLPool()

# FastAPI依赖注入
async def get_db_connection() -> AsyncGenerator[aiomysql.Connection, None]:
    """依赖注入：获取数据库连接"""
    connection = await mysql_pool.get_connection()
    try:
        yield connection
    finally:
        await mysql_pool.release_connection(connection)