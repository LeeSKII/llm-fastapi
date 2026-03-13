import asyncio
import logging
from typing import Optional, AsyncGenerator, Dict, Any, List
import dmPython
from ..core.config import settings

class DMPool:
    """达梦数据库连接池管理类"""
    
    def __init__(self):
        self.pool: Optional[asyncio.Queue] = None
        self.semaphore: Optional[asyncio.Semaphore] = None
        self._initialized = False
        self._init_error: Optional[str] = None
        self._lock = asyncio.Lock()
    
    async def create_pool(self) -> bool:
        """
        创建达梦数据库连接池
        返回: True表示成功，False表示失败
        """
        async with self._lock:
            if self._initialized:
                return True

            try:
                # 创建连接队列和信号量
                self.pool = asyncio.Queue(maxsize=settings["DM_POOL_MAXSIZE"])
                self.semaphore = asyncio.Semaphore(settings["DM_POOL_MAXSIZE"])

                # 预创建最小连接数
                for _ in range(settings["DM_POOL_MINSIZE"]):
                    conn = await self._create_connection()
                    await self.pool.put(conn)

                self._initialized = True
                self._init_error = None
                logging.info("✅ 达梦数据库连接池创建成功")
                logging.info(f"📊 达梦连接池配置: min={settings['DM_POOL_MINSIZE']}, max={settings['DM_POOL_MAXSIZE']}")
                return True

            except Exception as e:
                self._initialized = False
                self._init_error = str(e)
                logging.warning(f"⚠️ 创建达梦数据库连接池失败，服务将以降级模式运行: {e}")
                return False

    def get_init_error(self) -> Optional[str]:
        """获取初始化错误信息"""
        return self._init_error
    
    async def _create_connection(self) -> dmPython.connect:
        """创建单个数据库连接"""
        try:
            # 构建连接参数
            connect_params = {
                "user": settings["DM_USER"],
                "password": settings["DM_PASSWORD"],
                "server": settings["DM_HOST"],
                "port": settings["DM_PORT"]
            }
            
            # 只有在明确指定了 schema 时才添加该参数
            if settings["DM_SCHEMA"] and settings["DM_SCHEMA"].strip():
                connect_params["schema"] = settings["DM_SCHEMA"].strip()
                logging.info(f"连接到达梦数据库，使用指定 schema: {settings['DM_SCHEMA']}")
            else:
                logging.info(f"连接到达梦数据库，使用默认 schema")
            
            conn = dmPython.connect(**connect_params)
            
            # 如果没有指定 schema，可以尝试设置当前 schema
            if not settings["DM_SCHEMA"] or not settings["DM_SCHEMA"].strip():
                try:
                    # 尝试设置当前用户的默认 schema
                    cursor = conn.cursor()
                    cursor.execute(f"SET SCHEMA {settings['DM_USER']}")
                    cursor.close()
                    logging.info(f"已设置默认 schema: {settings['DM_USER']}")
                except Exception as schema_error:
                    logging.warning(f"设置默认 schema 失败，将使用连接默认值: {schema_error}")
            
            return conn
        except Exception as e:
            logging.error(f"❌ 创建达梦数据库连接失败: {e}")
            raise
    
    async def close_pool(self):
        """关闭连接池"""
        if not self._initialized:
            return
            
        try:
            # 关闭所有连接
            while not self.pool.empty():
                conn = await self.pool.get()
                try:
                    conn.close()
                except:
                    pass
            
            self._initialized = False
            logging.info("✅ 达梦数据库连接池已关闭")
        except Exception as e:
            logging.error(f"❌ 关闭达梦数据库连接池失败: {e}")
    
    async def get_connection(self) -> dmPython.connect:
        """从连接池获取连接"""
        if not self._initialized:
            raise Exception("达梦数据库连接池未初始化")
        
        async with self.semaphore:
            try:
                # 尝试从池中获取连接
                if not self.pool.empty():
                    conn = await self.pool.get()
                    # 检查连接是否有效
                    if await self._is_connection_valid(conn):
                        return conn
                    else:
                        # 连接无效，关闭并创建新连接
                        try:
                            conn.close()
                        except:
                            pass
                        conn = await self._create_connection()
                        return conn
                else:
                    # 池为空，创建新连接
                    return await self._create_connection()
            except Exception as e:
                logging.error(f"❌ 获取达梦数据库连接失败: {e}")
                raise
    
    async def release_connection(self, connection: dmPython.connect):
        """释放连接回连接池"""
        if not self._initialized or not connection:
            return
        
        try:
            # 检查连接是否有效
            if await self._is_connection_valid(connection):
                await self.pool.put(connection)
            else:
                # 连接无效，关闭它
                try:
                    connection.close()
                except:
                    pass
        except Exception as e:
            logging.error(f"❌ 释放达梦数据库连接失败: {e}")
            try:
                connection.close()
            except:
                pass
    
    async def _is_connection_valid(self, connection: dmPython.connect) -> bool:
        """检查连接是否有效"""
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except:
            return False
    
    async def execute(self, query: str, *args) -> int:
        """执行SQL语句，返回影响的行数"""
        conn = await self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, args)
            affected_rows = cursor.rowcount
            conn.commit()
            cursor.close()
            return affected_rows
        finally:
            await self.release_connection(conn)
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """查询单条记录"""
        conn = await self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, args)
            result = cursor.fetchone()
            if result:
                # 获取列名并构建字典
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, result))
            return None
        finally:
            await self.release_connection(conn)
    
    async def fetch_all(self, query: str, *args) -> List[Dict[str, Any]]:
        """查询多条记录"""
        conn = await self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, args)
            results = cursor.fetchall()
            if results:
                # 获取列名并构建字典列表
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in results]
            return []
        finally:
            await self.release_connection(conn)

# 全局达梦数据库实例
dm_pool = DMPool()

# FastAPI依赖注入
async def get_dm_db() -> AsyncGenerator[dmPython.connect, None]:
    """依赖注入：获取达梦数据库连接"""
    connection = await dm_pool.get_connection()
    try:
        yield connection
    finally:
        await dm_pool.release_connection(connection)