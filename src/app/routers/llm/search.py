import stat
import time
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage,AnyMessage,messages_to_dict,messages_from_dict,message_to_dict
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from enum import Enum
import os
import logging
from typing import AsyncGenerator, NotRequired
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
from tavily import TavilyClient
from dotenv import load_dotenv

# 环境变量加载
load_dotenv()

router = APIRouter()

api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("SELF_HOST_URL")
model_name = "Qwen3-235B"

# base_url = os.getenv("QWEN_API_BASE_URL")
# model_name = "qwen-plus-latest"
# Initialize Tavily client
tavily_api_key = os.getenv('TAVILY_API_KEY')
tavily_client = TavilyClient(api_key=tavily_api_key)

llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0.01)

reply_system_prompt = "你是一位乐于助人的助手。no_think" 

judge_need_web_search_system_prompt = "根据用户提出的问题:\n{query}\n。综合上下文信息，判断是否有足够的信息做出回答，输出json格式数据，严格遵循json格式：\n{format_instructions}"

# 判断是否需要网页搜索
class WebSearchJudgement(BaseModel):
    isNeedWebSearch: bool = Field(description="是否需要通过网页搜索获取足够的信息进行回复")
    reason: str = Field(description="选择执行该动作的原因")
    confidence: float = Field(description="置信度，评估是否需要网页搜索的可靠性")

class SearchDepthEnum(str, Enum):
    BASIC = "basic"
    ADVANCED = "advanced"

class WebSearchQuery(BaseModel):
    query: str = Field(description="预备进行网络搜索查询的问题")
    search_depth:SearchDepthEnum = Field(description="搜索的深度，枚举值：BASIC、ADVANCED")
    reason: str = Field(description="生成该搜索问题的原因")
    confidence: float = Field(description="关联度，评估生成的搜索问题和用户提问的关联度")

# 定义状态类
class OverallState(TypedDict):
    query:str
    web_search_query:str
    web_search_depth:str
    web_search_results:dict
    response:str
    messages: list[dict]
    isNeedWebSearch: bool
    reason: str
    confidence: float

def analyze_need_web_search(state: OverallState)-> OverallState:
    """判断是否需要进行网页搜索"""
    parser = PydanticOutputParser(pydantic_object=WebSearchJudgement)
    # 获取 JSON Schema 的格式化指令
    format_instructions = parser.get_format_instructions()
    prompt_template = PromptTemplate.from_template(judge_need_web_search_system_prompt)
    prompt = prompt_template.format(query=state['query'],format_instructions=format_instructions)
    
    response = llm.invoke([{'role':'system','content':reply_system_prompt},*state['messages'],{"role":"user","content":prompt}])

    model = parser.parse(response.content)

    logging.info(f"Parsed analyze_need_web_search model: {model}")

    return {"query":state['query'],"messages":state['messages'],"isNeedWebSearch":model.isNeedWebSearch,"reason":model.reason,"confidence":model.confidence}

def generate_search_query(state: OverallState)-> OverallState:
    """生成搜索查询"""
    query = state['query']
    messages = state.get("messages", [])
    parser = PydanticOutputParser(pydantic_object=WebSearchQuery)
    # 获取 JSON Schema 的格式化指令
    format_instructions = parser.get_format_instructions()
    prompt = f"根据用户的问题：\n{query},以及上下文的messages生成一个合适的网络搜索查询。使用json结构化输出，严格遵循的schema：\n{format_instructions}"

    response = llm.invoke([{'role':'system','content':reply_system_prompt},*messages,{"role":"user","content":prompt}])

    model = parser.parse(response.content)

    logging.info(f"Parsed generate_search_query model: {model}")

    return {"web_search_query":model.query,"web_search_depth":model.search_depth,"reason":model.reason,"confidence":model.confidence}


def web_search(state: OverallState)-> OverallState:
    """网页搜索"""
    query = state['web_search_query']
    search_depth = state['web_search_depth']
    messages = state.get("messages", [])
    search_result = tavily_client.search(query,search_depth=search_depth)
    # 如果这里包含了langchain提供的message类型，那么会直接触发message的流式更新动作
    return {"web_search_results":search_result['results'],"messages":messages}

def assistant_node(state: OverallState) -> OverallState:
    """助手响应"""
    if(state['isNeedWebSearch']):
        send_messages = [{'role':'system','content':reply_system_prompt},*state['messages'],{"role":"user","content":f"搜索结果：{state['web_search_results']}，用户提问：{state['query']}"}]
    else:
        send_messages = [{'role':'system','content':reply_system_prompt},*state['messages'],{"role":"user","content":f"{state['query']}"}]

    ai_response = llm.invoke(send_messages)
    messages = [*state["messages"],{"role":"user","content":f"{state['query']}"},{"role":"assistant","content":ai_response.content}]
    return {"response":ai_response.content,"messages":messages}

def need_web_search(state: OverallState)->bool:
    """判断是否需要网页搜索"""
    return state['isNeedWebSearch']

# 创建图形
workflow = StateGraph(OverallState)

# 添加节点
workflow.add_node("analyze_need_web_search", analyze_need_web_search)
workflow.add_node("generate_search_query", generate_search_query)
workflow.add_node("web_search", web_search)
workflow.add_node("assistant", assistant_node)


# 添加普通边
workflow.add_edge(START,"analyze_need_web_search")
workflow.add_conditional_edges("analyze_need_web_search",need_web_search,{True: "generate_search_query", False: "assistant"})
workflow.add_edge("generate_search_query","web_search")
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
async def run_workflow_non_stream(query: str):
    result = await app.ainvoke({"query": query})
    return result

class InputData(TypedDict):
    query: str  # 必填字段
    messages: NotRequired[list[dict]]  # 可选字段

# LLM stream传输
@router.post("/stream", tags=["search"])
async def run_workflow_stream(input_data: InputData):
    query = input_data["query"]  # 必填字段直接访问
    messages = input_data.get("messages", [])

    logging.info(f"Query: {query}")
    logging.info(f"Messages: {messages}")
    
    if not query:
        # 使用HTTP异常更符合REST规范
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    async def stream_updates() -> AsyncGenerator[str, None]:
        try:
            # 添加心跳机制 (每30秒发送空注释)
            last_sent = time.time()
            
            async for chunk in app.astream({"query": query,"messages":messages}, stream_mode=["updates","messages"]):
                logging.info(f"Chunk: {chunk}")
                mode,*_ = chunk
                
                # 发送心跳 (防止代理超时断开)
                if time.time() - last_sent > 30:
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
