from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("SELF_HOST_URL")
model_name = "qwen-plus-latest"
model_name = "Qwen3-235B"

llm = ChatOpenAI(model=model_name,api_key=api_key,base_url=base_url,temperature=0.01)

# 定义 LangGraph 状态
class State(TypedDict):
    topic: str
    result:str

# 节点1：根据主题生成内容
async def llm_by_topic(state: State) -> State:
    response = await llm.ainvoke([{"role":"system","content":"使用日语回答，生成500字以上的内容,no_think"},{"role":"user","content":state["topic"]}])
    return {"result": response.content}


# 构建 LangGraph 工作流
workflow = StateGraph(State)
workflow.add_node("llm_by_topic", llm_by_topic)
workflow.add_edge(START, "llm_by_topic")
workflow.add_edge("llm_by_topic", END)

# 编译工作流
graph = workflow.compile()