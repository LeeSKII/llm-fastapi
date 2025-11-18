from ...utils import logger
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(TypedDict):
    APP_NAME: str
    LIFESPAN: object

    # MySQL 数据库配置
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DB: str = ""
    MYSQL_CHARSET: str = "utf8mb4"
    MYSQL_POOL_MINSIZE: int = 2
    MYSQL_POOL_MAXSIZE: int = 10
    MYSQL_POOL_RECYCLE: int = 3600  # 连接回收时间(秒)
    
    # 达梦数据库配置
    DM_HOST: str = "localhost"
    DM_PORT: int = 5236
    DM_USER: str = "SYSDBA"
    DM_PASSWORD: str = "SYSDBA001"
    DM_SCHEMA: str = ""
    DM_POOL_MINSIZE: int = 2
    DM_POOL_MAXSIZE: int = 10
    DM_POOL_RECYCLE: int = 3600  # 连接回收时间(秒)
    

def get_settings() -> Settings:
    return {
        "APP_NAME": os.getenv("APP_NAME", "LLM-FastAPI"),
        
        # MySQL配置
        "MYSQL_HOST": os.getenv("MYSQL_HOST", "localhost"),
        "MYSQL_PORT": int(os.getenv("MYSQL_PORT", 3306)),
        "MYSQL_USER": os.getenv("MYSQL_USER", "root"),
        "MYSQL_PASSWORD": os.getenv("MYSQL_PASSWORD", ""),
        "MYSQL_CHARSET": os.getenv("MYSQL_CHARSET", "utf8mb4"),
        "MYSQL_DB": os.getenv("MYSQL_DB", ""),
        "MYSQL_POOL_MINSIZE": int(os.getenv("MYSQL_POOL_MINSIZE", "2")),
        "MYSQL_POOL_MAXSIZE": int(os.getenv("MYSQL_POOL_MAXSIZE", "10")),
        "MYSQL_POOL_RECYCLE": int(os.getenv("MYSQL_POOL_RECYCLE", "3600")),
        
        # 达梦数据库配置
        "DM_HOST": os.getenv("DM_HOST", "localhost"),
        "DM_PORT": int(os.getenv("DM_PORT", 5236)),
        "DM_USER": os.getenv("DM_USER", "SYSDBA"),
        "DM_PASSWORD": os.getenv("DM_PASSWORD", "SYSDBA001"),
        "DM_SCHEMA": os.getenv("DM_SCHEMA", ""),
        "DM_POOL_MINSIZE": int(os.getenv("DM_POOL_MINSIZE", "2")),
        "DM_POOL_MAXSIZE": int(os.getenv("DM_POOL_MAXSIZE", "10")),
        "DM_POOL_RECYCLE": int(os.getenv("DM_POOL_RECYCLE", "3600")),
    }

settings = get_settings()