import re
from agno.agent import Agent
from agno.models.openai import OpenAILike
import datetime
import httpx
import json
import os
from typing import Optional, Any

def get_last_week_timestamps():
    # 获取当前时间
    now = datetime.datetime.now()
    
    # 计算今天是本周的第几天（周一为0，周日为6）
    today_weekday = now.weekday()
    
    # 计算上周日的日期（今天的日期 - 今天星期几 - 1天）
    last_sunday = now - datetime.timedelta(days=today_weekday + 1)
    
    # 计算上周一的日期（上周日 - 6天）
    last_monday = last_sunday - datetime.timedelta(days=6)
    
    # 设置上周一的时间为00:00:00
    last_monday_start = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # 设置上周日的时间为23:59:59.999999
    last_sunday_end = last_sunday.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    # 转换为时间戳（秒级），然后乘以1000得到13位毫秒级时间戳
    monday_timestamp = int(last_monday_start.timestamp() * 1000)
    sunday_timestamp = int(last_sunday_end.timestamp() * 1000)
    
    # 格式化日期为YYYY-MM-DD
    monday_date_str = last_monday.strftime("%Y-%m-%d")
    sunday_date_str = last_sunday.strftime("%Y-%m-%d")
    
    return monday_timestamp, sunday_timestamp,monday_date_str,sunday_date_str

async def get_access_token(appKey: str, appSecret: str) -> str | None:
    """异步获取钉钉access_token"""
    url = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
    json_data = {
        "appKey": appKey,
        "appSecret": appSecret
    }
    
    async with httpx.AsyncClient() as client:  # 使用异步客户端
        try:
            response = await client.post(
                url,
                json=json_data,
                timeout=10.0
            )
            response.raise_for_status()  # 自动处理4xx/5xx错误
            return response.json()["accessToken"]
            
        except httpx.HTTPStatusError as e:
            print(f"HTTP错误: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"其他错误: {str(e)}")
        return None

async def get_report(
    access_token: str, 
    start_time: int, 
    end_time: int
) -> Optional[str]:
    """异步获取钉钉工作周报数据"""
    url = f"https://oapi.dingtalk.com/topapi/report/list?access_token={access_token}"
    
    json_data = {
        "cursor": "0",
        "start_time": start_time,
        "template_name": "管理数字化工作周报",
        "size": 20,
        "end_time": end_time
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                json=json_data,
                timeout=10.0
            )
            response.raise_for_status()  # 自动处理4xx/5xx错误
            return json.dumps(response.json(), ensure_ascii=False)
            
        except httpx.HTTPStatusError as e:
            print(f"API请求失败 [HTTP {e.response.status_code}]: {e.response.text}")
        except json.JSONDecodeError:
            print("响应数据JSON解析失败")
        except Exception as e:
            print(f"未知错误: {str(e)}")
        return None
    

async def post_report_with_llm():
    local_settings = {
        'api_key' : '123',
        'base_url' : 'http://192.168.0.166:8000/v1',
        'id' : 'Qwen3-235B'
    }
    ding_app_key = os.getenv('DING_APP_KEY')
    ding_app_secret = os.getenv('DING_APP_SECRET')
    monday_ts, sunday_ts,monday_date_str,sunday_date_str = get_last_week_timestamps()
    access_token = await get_access_token(appKey=ding_app_key, appSecret=ding_app_secret)
    weekly_report = await get_report(access_token, monday_ts, sunday_ts)
    agent = Agent(model=OpenAILike(**local_settings),description='你是一位周报总结专家.',instructions=['首先按照用户进行分组','根据用户的项目进行分组汇总项目事项','根据项目信息最后进行简短\客观\事实的分析总结','不要遗漏任何项目和项目事项信息',"严谨虚构和假设任何数据,没有请回答无记录",f'报告标题为:{monday_date_str}-{sunday_date_str}周报'],add_datetime_to_instructions=True,markdown=True,telemetry=False)
    response = await agent.arun(message=weekly_report)
    content = response.content
    return content