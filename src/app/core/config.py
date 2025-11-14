from ...utils import logger
from ..scheduler.start import lifespan
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
    
    # 连接池配置
    DB_POOL_MINSIZE: int = 1
    DB_POOL_MAXSIZE: int = 20
    DB_POOL_RECYCLE: int = 3600  # 连接回收时间(秒)

def get_settings() -> Settings:
    return {
        "APP_NAME": os.getenv("APP_NAME", "LLM-FastAPI"),
        "MYSQL_HOST": os.getenv("MYSQL_HOST", "localhost"),
        "MYSQL_PORT": int(os.getenv("MYSQL_PORT", 3306)),
        "MYSQL_USER": os.getenv("MYSQL_USER", "root"),
        "MYSQL_PASSWORD": os.getenv("MYSQL_PASSWORD", ""),
        "MYSQL_DB": os.getenv("MYSQL_DB", ""),
        "MYSQL_POOL_MINSIZE": int(os.getenv("MYSQL_POOL_MINSIZE", "2")),
        "MYSQL_POOL_MAXSIZE": int(os.getenv("MYSQL_POOL_MAXSIZE", "10")),
        "MYSQL_POOL_RECYCLE": int(os.getenv("MYSQL_POOL_RECYCLE", "3600")),
        
        "LIFESPAN": lifespan
    }

settings = get_settings()