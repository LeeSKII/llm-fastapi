import aiomysql
from typing import Optional, AsyncGenerator
from ..core.config import settings
import logging

class MySQLPool:
    """MySQL连接池管理类"""
    
    def __init__(self):
        self.pool: Optional[aiomysql.Pool] = None
    
    async def create_pool(self):
        """创建MySQL连接池"""
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
            logging.info("✅ MySQL连接池创建成功")
            logging.info(f"📊 MySQL连接池配置: min={settings['MYSQL_POOL_MINSIZE']}, max={settings['MYSQL_POOL_MAXSIZE']}")
        except Exception as e:
            logging.error(f"❌ 创建MySQL连接池失败: {e}")
            raise
    
    async def close_pool(self):
        """关闭连接池"""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            logging.info("✅ MySQL连接池已关闭")
    
    async def get_connection(self) -> aiomysql.Connection:
        """从连接池获取连接"""
        if not self.pool:
            raise Exception("MySQL连接池未初始化")
        return await self.pool.acquire()
    
    async def release_connection(self, connection: aiomysql.Connection):
        """释放连接回连接池"""
        if self.pool and connection:
            await self.pool.release(connection)
    
    async def execute(self, query: str, *args) -> int:
        """执行SQL语句，返回影响的行数"""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, args)
                return cur.rowcount
    
    async def fetch_one(self, query: str, *args) -> Optional[dict]:
        """查询单条记录"""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, args)
                result = await cur.fetchone()
                return dict(result) if result else None
    
    async def fetch_all(self, query: str, *args) -> list:
        """查询多条记录"""
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