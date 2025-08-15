import time
import logging
from typing import AsyncGenerator, Literal,List,Optional
from langchain_core.messages import message_to_dict
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel, Field
import asyncio

from ....utils.llm_graph import contract_graph as app

router = APIRouter()

class ContractSearch(BaseModel):
    query: str = Field(..., description="搜索关键字")
    messages:Optional[List[dict]] = Field(None, description="用户消息")

# 测试接口
@router.get("/",tags=["contract"])
async def test():
    return {"test":"contract"}

@router.post("/stream", tags=["contract"])
async def run_workflow_stream(input_data:ContractSearch,request: Request):
    """
    运行流式工作流
    
    Args:
        request: 输入数据
            
    Returns:
        StreamingResponse: SSE流式响应对象
    """
    logging.info(f"开始请求，数据体: {input_data}")
    messages = input_data.messages or []
    query = input_data.query
    
    # 由于中断由API入口介入，持久化数据的过程应该控制在这里，可以由custom自定义事件进行控制
    async def stream_updates(req: Request) -> AsyncGenerator[str, None]:
        try:
            logging.info(f"开始流式传输:")
            # 添加心跳机制 (每30秒发送空注释)
            last_sent = time.time()
            heartbeat_interval = 30
            
            async for chunk in app.graph.astream(
                {
                    "query": query,"messages":messages
                    
                }, 
                stream_mode=["messages", "updates", "custom"]
            ):
                # --- 在循环开始时主动检查连接状态 ---
                if await req.is_disconnected():
                    logging.warning("客户端在流式传输过程中断开连接，提前终止。")
                    break # 退出循环
                
                logging.info(f"Chunk: {chunk}")
                mode, *_ = chunk
                
                # 发送心跳 (防止代理超时断开)
                if time.time() - last_sent > heartbeat_interval:
                    yield ":keep-alive\n\n"
                    last_sent = time.time()
                
                if mode == "updates":
                    mode, data = chunk
                    node_name = list(data.keys())[0]
                    # 结构化响应数据
                    response = {
                        "mode": mode,
                        "node": node_name,
                        "data": data[node_name]
                    }
                    yield f"event: updates\ndata: {json.dumps(response)}\n\n"
                    last_sent = time.time()
                
                elif mode == "messages":
                    mode, message_chunk = chunk
                    llm_token, metadata = message_chunk
                    # 结构化响应数据
                    response = {
                        "mode": mode,
                        "node": metadata.get('langgraph_node', ""),
                        "data": message_to_dict(llm_token),
                    }
                    yield f"event: messages\ndata: {json.dumps(response)}\n\n"
                    last_sent = time.time()
                
                # 自定义消息用来显示当前正在运行的节点
                elif mode == "custom":
                    mode, data = chunk
                    node_name = data.get('node', "")
                    # 结构化响应数据
                    response = {
                        "mode": mode,
                        "node": node_name,
                        "data": data
                    }
                    yield f"event: custom\ndata: {json.dumps(response)}\n\n"
                    last_sent = time.time()
            
        except asyncio.CancelledError:
            # 这是捕获客户端中断的核心位置
            logging.warning("检测到客户端连接中断 (CancelledError)，流已终止。")
            # 此处不需要 yield 任何东西，因为连接已经关闭
            # 可以在这里执行任何必要的资源清理操作
            
        except Exception as e:
            logging.error(f"流式传输错误, 错误: {str(e)}", exc_info=True)
            # 发送错误信息而不是直接断开
            error_msg = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_msg}\n\n"
            logging.error(f"Streaming error: {str(e)}")
        
        finally:
            logging.info(f"流式传输结束:")
            # 发送结束事件
            yield "event: end\ndata: {}\n\n"
    
    return StreamingResponse(
        stream_updates(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            # 添加浏览器兼容头部
            "Content-Encoding": "none",
            "X-SSE-Content-Type": "text/event-stream"
        }
    )
