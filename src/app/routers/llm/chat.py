import time
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage,AnyMessage
from langgraph.graph.message import add_messages
import os
import logging
from typing import AsyncGenerator, Literal
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json

router = APIRouter()

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("SELF_HOST_URL")
model_name = "Qwen3-235B"

llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0.01)


# 定义状态类
class ChatState(MessagesState):
    messages: list[AnyMessage]

def assistant_node(state: ChatState) -> ChatState:
    """助手响应"""
    ai_response = llm.invoke(state['messages'])
    return {"messages": [*state['messages'],AIMessage(content=ai_response.content)]}

# 创建图形
workflow = StateGraph(ChatState)

# 添加节点
workflow.add_node("assistant", assistant_node)


# 添加普通边
workflow.add_edge(START,"assistant")
workflow.add_edge("assistant",END)

# 编译图形
app = workflow.compile()

# 测试接口
@router.get("/",tags=["chat"])
async def run_workflow(input_data: dict):
    return {"result":"chat"}

# LLM value传输
@router.post("/messages",tags=["chat"])
async def run_workflow(input_data: dict):
    messages = input_data.get("messages", "")  
    if not messages:
        return {"result": "Messages is empty"}
    result = await app.ainvoke({"messages": messages})
    return result

# LLM stream传输
@router.post("/stream", tags=["chat"])
async def run_workflow(input_data: dict):
    messages = input_data.get("messages", [])
    
    if not messages:
        # 使用HTTP异常更符合REST规范
        raise HTTPException(status_code=400, detail="Messages cannot be empty")
    
    async def stream_updates() -> AsyncGenerator[str, None]:
        try:
            input_state = {"messages": messages}
            
            # 添加心跳机制 (每15秒发送空注释)
            last_sent = time.time()
            
            async for chunk in app.astream(input_state, stream_mode="messages"):
                (message_chunk, metadata) = chunk
                
                # 发送心跳 (防止代理超时断开)
                if time.time() - last_sent > 15:
                    yield ":keep-alive\n\n"
                    last_sent = time.time()
                
                if message_chunk.content:
                    # 结构化响应数据
                    response = {
                        "content": message_chunk.content,
                        "metadata": {
                            "type": getattr(message_chunk, "type", "text"),
                            "role": getattr(message_chunk, "role", "assistant")
                        }
                    }
                    yield f"data: {json.dumps(response)}\n\n"
                    last_sent = time.time()
        
        except Exception as e:
            # 发送错误信息而不是直接断开
            error_msg = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_msg}\n\n"
            logging.error(f"Streaming error: {str(e)}")
        
        finally:
            # 发送结束事件
            yield "event: end\ndata: {}\n\n"
    
    return StreamingResponse(
        stream_updates(),
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
