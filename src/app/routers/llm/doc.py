from fastapi import APIRouter
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import asyncio

from ....utils.llm_graph import doc_audit

router = APIRouter()

@router.get("/", tags=["doc"])
async def test():
    return {"message": "Hello, doc!"}


# HTTP 端点：流式返回节点更新
@router.get("/stream/{topic}",tags=["doc"])
async def stream_workflow(topic: str) -> StreamingResponse:
    async def stream_updates() -> AsyncGenerator[str, None]:
        input_state = {"topic": topic}
        async for (message_chunk, metadata) in doc_audit.graph.astream(input_state, stream_mode="messages"):
            # yield f"data: {chunk}\n\n"
            # 提取 content
            content = message_chunk.content          
            # 只发送有内容的块（过滤空内容）
            if content:  
                yield f"{content}"
                # await asyncio.sleep(0.1)  # 控制流速度
    
    return StreamingResponse(
        stream_updates(),
        media_type="text/event-stream",
         # X-Accel-Buffering 解决Nginx代理缓冲问题
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive","X-Accel-Buffering": "no"}
    )

# HTTP 端点：流式返回节点更新
@router.post("/stream",tags=["doc"])
async def stream_workflow(input_data: dict) -> StreamingResponse:
    content = input_data.get("content", "")
    async def stream_updates(content_str: str) -> AsyncGenerator[str, None]:
        input_state = {"content": content_str}
        print(input_state)
        async for (message_chunk, metadata) in doc_audit.graph.astream(input_state, stream_mode="messages"):
            # yield f"data: {chunk}\n\n"
            # 提取 content
            content = message_chunk.content          
            # 只发送有内容的块（过滤空内容）
            if content:  
                # SSE格式强制要求 必须包含 data: 前缀和双换行符 \n\n
                yield f"data: {content}\n\n"
                # await asyncio.sleep(0.1)  # 控制流速度
    
    return StreamingResponse(
        stream_updates(content),
        media_type="text/event-stream",
         # X-Accel-Buffering 解决Nginx代理缓冲问题
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive","X-Accel-Buffering": "no"}
    )

# WebSocket 端点：实时推送工作流更新
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # 接收客户端输入
        data = await websocket.receive_json()
        topic = data.get("topic", "default")
        input_state = {"topic": topic}
        
        # 流式推送 LangGraph 输出
        async for chunk in doc_audit.graph.astream(input_state, stream_mode="updates"):
            await websocket.send_json(chunk)
            await asyncio.sleep(0.1)  # 模拟延迟，防止过快发送
        
        await websocket.send_json({"status": "completed"})
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

# 普通 HTTP 端点：返回最终结果
@router.post("/arun")
async def run_workflow(input_data: dict):
    content = input_data.get("content", "")
    if not content:
        return {"result": "Content is empty"}
    result = await doc_audit.graph.ainvoke({"content": content})
    return result