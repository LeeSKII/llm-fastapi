from pyexpat.errors import messages
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

base_url = os.getenv("QWEN_API_BASE_URL")
model_name = "qwen-plus-latest"
# Initialize Tavily client
tavily_api_key = os.getenv('TAVILY_API_KEY')
tavily_client = TavilyClient(api_key=tavily_api_key)

llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0.01)

system_prompt = "你是一位老学究，回复时使用文言文，显得高级。no_think" 

# 定义状态类
class OverallState(TypedDict):
    query:str
    web_search:dict
    response:str
    messages: list[dict]
    
class WebSearchState(TypedDict):
    query:str
    messages:list[dict]
    web_search:dict

def web_search(state: OverallState)-> WebSearchState:
    """网页搜索"""
    query = state['query']
    messages = state.get("messages", [])
    search_result = tavily_client.search(query)
    # 如果这里包含了langchain提供的message类型，那么会直接触发message的流式更新动作
    return {"web_search":search_result['results'],"messages":messages}

def assistant_node(state: WebSearchState) -> OverallState:
    """助手响应"""
    ai_response = llm.invoke([{'role':'system','content':system_prompt},*state['messages'],{"role":"user","content":f"搜索结果：{state['web_search']}，用户提问：{state['query']}"}])
    messages = [*state["messages"],{"role":"user","content":f"{state['query']}"},{"role":"assistant","content":ai_response.content}]
    return {"response":ai_response.content,"messages":messages}

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
    messages = input_data.get("messages", [])
    
    if not query:
        # 使用HTTP异常更符合REST规范
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    async def stream_updates() -> AsyncGenerator[str, None]:
        try:
            # 添加心跳机制 (每15秒发送空注释)
            last_sent = time.time()
            
            async for chunk in app.astream({"query": query,"messages":messages}, stream_mode=["updates","messages"]):
                logging.info(f"Chunk: {chunk}")
                mode,*_ = chunk
                
                # 发送心跳 (防止代理超时断开)
                if time.time() - last_sent > 15:
                    yield ":keep-alive\n\n"
                    last_sent = time.time()
                
                if mode == "updates":
                    mode,data = chunk
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
                    mode,message_chunk = chunk
                    llm_token,metadata = message_chunk
                    # 结构化响应数据
                    response = {
                        "mode": mode,
                        "llm_token": message_to_dict(llm_token),
                        "metadata": metadata
                    }
                    yield f"event: messages\ndata: {json.dumps(response)}\n\n"
                
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
