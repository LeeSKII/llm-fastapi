import time
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage,AnyMessage,messages_to_dict,messages_from_dict,message_to_dict
from langgraph.graph.message import add_messages
import os
import logging
from typing import AsyncGenerator, Literal
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
from tavily import TavilyClient

router = APIRouter()

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("SELF_HOST_URL")
model_name = "Qwen3-235B"
# Initialize Tavily client
tavily_api_key = os.getenv('TAVILY_API_KEY')
tavily_client = TavilyClient(api_key=tavily_api_key)

llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0.01)


# 定义状态类
class OverallState(TypedDict):
    query:str
    web_search:dict
    response:str
    messages: list[AnyMessage]

def web_search(state: OverallState):
    """网页搜索"""
    query = state['query']
    search_result = tavily_client.search(query)
    search_result_str = json.dumps(search_result['results'])
    return {"web_search":search_result}

def assistant_node(state: OverallState) -> OverallState:
    """助手响应"""
    ai_response = llm.invoke([HumanMessage(content=f"no_think，根据搜索背景知识{state['web_search']}，回答用户提出的问题：{state['query']}")])
    return {"response":ai_response.content}

# 创建图形
workflow = StateGraph(OverallState)

# 添加节点
workflow.add_node("web_search", web_search)
workflow.add_node("assistant", assistant_node)


# 添加普通边
workflow.add_edge(START,"web_search")
workflow.add_edge("web_search","assistant")
workflow.add_edge("assistant",END)

# 编译图形
app = workflow.compile()

# 测试接口
@router.get("/{query}",tags=["search"])
async def test(query: str):
    return {"result":query}

# LLM value传输
@router.get("/query/{query}",tags=["search"])
async def run_workflow(query: str):
    result = await app.ainvoke({"query": query})
    return result

# LLM stream传输
@router.post("/stream", tags=["search"])
async def run_workflow(input_data: dict):
    query = input_data.get("query", "")
    
    if not query:
        # 使用HTTP异常更符合REST规范
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    async def stream_updates() -> AsyncGenerator[str, None]:
        try:
            # 添加心跳机制 (每15秒发送空注释)
            last_sent = time.time()
            
            async for chunk in app.astream({"query": query}, stream_mode=["updates","messages"]):
                logging.info(f"Chunk: {chunk}")
                mode,*_ = chunk
                
                # 发送心跳 (防止代理超时断开)
                if time.time() - last_sent > 15:
                    yield ":keep-alive\n\n"
                    last_sent = time.time()
                
                if mode == "updates":
                    mode,data = chunk
                    # 结构化响应数据
                    response = {
                        "mode": mode,
                        "llm_token": data
                    }
                    yield f"data: {json.dumps(response)}\n\n"
                    last_sent = time.time()

                elif mode == "messages":
                    mode,message_chunk = chunk
                    llm_token,metadata = message_chunk
                    # 结构化响应数据
                    response = {
                        "mode": mode,
                        "llm_token": message_to_dict(llm_token),
                        "metadata": metadata
                    }
                    yield f"data: {json.dumps(response)}\n\n"
                
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
