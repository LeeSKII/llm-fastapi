import os
import logging
from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field
import lancedb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

CONTRACT_DB_PATH = os.getenv("CONTRACT_DB_PATH")


class KeywordSearchRequest(BaseModel):
    keywords: List[str] = Field(..., description="搜索关键字列表")
    limit: Optional[int] = Field(20, description="每个关键字返回的最大结果数")


class ContractInfo(BaseModel):
    contact_no: str
    project_name: str
    date: str
    contract_meta: Optional[str] = None
    equipment_table: Optional[str] = None


class KeywordSearchResponse(BaseModel):
    total: int
    results: List[dict]


@router.get("/", tags=["contract_search"])
async def test():
    return {"message": "search service is ready"}


@router.post("/keyword", tags=["contract_search"], response_model=KeywordSearchResponse)
async def keyword_search(request: KeywordSearchRequest):
    """
    根据关键字搜索合同文件列表

    Args:
        request: 包含关键字列表和可选的limit参数

    Returns:
        KeywordSearchResponse: 包含搜索结果总数和合同列表
    """
    try:
        if not CONTRACT_DB_PATH:
            return {"total": 0, "results": [], "error": "CONTRACT_DB_PATH not configured"}

        df_list = []
        db = lancedb.connect(CONTRACT_DB_PATH)
        table = db.open_table("contract_table")

        for keyword in request.keywords:
            logging.info(f"Searching for keyword: {keyword}")
            search_results = table.search().where(f"doc LIKE '%{keyword}%'").limit(request.limit).to_pandas()
            logging.info(f"Found {len(search_results)} results for keyword: {keyword}")
            df_list.append(search_results)

        if not df_list:
            return {"total": 0, "results": []}

        # 合并所有搜索结果
        search_results = pd.concat(df_list, ignore_index=True)

        # 根据合同编号和项目名称去重
        final_df = search_results.drop_duplicates(subset=['contact_no', 'project_name'])

        logging.info(f"Total unique results: {len(final_df)}")

        # 调试：打印所有搜索到的文件信息
        logging.info(f"Columns: {final_df.columns.tolist()}")
        for idx, row in final_df.iterrows():
            logging.info(f"Result {idx}: {dict(row)}")

        # 转换为可JSON序列化的格式
        results = []
        for _, row in final_df.iterrows():
            record = {}
            for col in final_df.columns:
                val = row[col]
                # 处理NaN值 - 使用scalar检查避免数组问题
                try:
                    is_na = pd.isna(val)
                    if isinstance(is_na, bool) and is_na:
                        record[col] = None
                        continue
                except:
                    pass

                # 处理datetime类型
                if hasattr(val, 'isoformat'):
                    record[col] = val.isoformat()
                # 处理Decimal等其他类型
                else:
                    record[col] = str(val) if not isinstance(val, (str, int, float, bool, type(None))) else val
            results.append(record)

        return {
            "total": len(final_df),
            "results": results
        }

    except Exception as e:
        logging.error(f"Keyword search error: {str(e)}", exc_info=True)
        return {
            "total": 0,
            "results": [],
            "error": str(e)
        }

