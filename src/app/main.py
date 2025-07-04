from typing import Union

from fastapi import FastAPI

from .scheduler.start import lifespan
from .routers import reports
from ..utils import logger

from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, AsyncGenerator
import asyncio
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("QWEN_API_BASE_URL")

app = FastAPI(lifespan=lifespan)

app.include_router(reports.router,prefix="/reports")

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

llm = ChatOpenAI(model="qwen-plus-latest",api_key=api_key,base_url=base_url,temperature=0.01)

# 定义 LangGraph 状态
class State(TypedDict):
    topic: str
    result:str

# 节点1：根据主题生成内容
async def llm_by_topic(state: State) -> State:
    response = await llm.ainvoke([{"role":"system","content":"使用日语回答，生成500字以上的内容"},{"role":"user","content":state["topic"]}])
    return {"result": response.content}


# 构建 LangGraph 工作流
workflow = StateGraph(State)
workflow.add_node("llm_by_topic", llm_by_topic)
workflow.add_edge(START, "llm_by_topic")
workflow.add_edge("llm_by_topic", END)

# 编译工作流
graph = workflow.compile()

# HTTP 端点：流式返回节点更新
@app.get("/stream/{topic}")
async def stream_workflow(topic: str) -> StreamingResponse:
    async def stream_updates() -> AsyncGenerator[str, None]:
        input_state = {"topic": topic}
        async for chunk in graph.astream(input_state, stream_mode="messages"):
            yield f"data: {chunk}\n\n"
    
    return StreamingResponse(
        stream_updates(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# WebSocket 端点：实时推送工作流更新
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # 接收客户端输入
        data = await websocket.receive_json()
        topic = data.get("topic", "default")
        input_state = {"topic": topic}
        
        # 流式推送 LangGraph 输出
        async for chunk in graph.astream(input_state, stream_mode="updates"):
            await websocket.send_json(chunk)
            await asyncio.sleep(0.1)  # 模拟延迟，防止过快发送
        
        await websocket.send_json({"status": "completed"})
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

# 普通 HTTP 端点：返回最终结果
@app.post("/run")
async def run_workflow(input_data: dict):
    topic = input_data.get("topic", "default")
    result = await graph.ainvoke({"topic": topic})
    return result