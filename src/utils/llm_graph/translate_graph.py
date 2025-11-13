from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv

load_dotenv()

qwen_api_key = os.getenv("QWEN_API_KEY")
qwen_base_url = os.getenv("QWEN_API_BASE_URL")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_base_url = os.getenv("DEEPSEEK_API_BASE_URL")
local_base_url = os.getenv("SELF_HOST_URL")

class Translation(BaseModel):
    input: str = Field(..., description="输入文本")
    reason: str = Field(..., description="翻译解释")
    text: str = Field(..., description="翻译后的文本")

system_prompt = """你是一位专业且严谨的工程设计领域翻译专家，擅长将用户输入的任何目标语言翻译成目标语言{target}，请按要求翻译每轮对话用户提供的输入。"""

# 定义 LangGraph 状态
class State(TypedDict):
    input: str
    target_language: str
    qwen_result: Translation
    deepseek_result: Translation
    local_result: Translation
    summary: str

# 节点1：Qwen
def llm_translation_qwen(state: State) -> State:
    try:
        model = ChatOpenAI(model="qwen3-max", api_key=qwen_api_key, base_url=qwen_base_url, temperature=0.01)
        agent = create_agent(
            model=model,
            tools=[],
            response_format=ToolStrategy(Translation)
        )
        
        result = agent.invoke({
            "messages": [
                {"role": "system", "content": system_prompt.format(target=state["target_language"])},
                {"role": "user", "content": state["input"]}
            ]
        })
        
        translation = result["structured_response"]
        return {"qwen_result": translation}
    except Exception as e:
        return {"qwen_result": Translation(
            text="千问模型调用失败，请检查服务是否正常",
            input=state["input"],
            reason=str(e)
        )}

# 节点2：Deepseek
def llm_translation_deepseek(state: State) -> State:
    try:
        model = ChatOpenAI(model="deepseek-chat", api_key=deepseek_api_key, base_url=deepseek_base_url, temperature=0.01)
        agent = create_agent(
            model=model,
            tools=[],
            response_format=ToolStrategy(Translation)
        )
        
        result = agent.invoke({
            "messages": [
                {"role": "system", "content": system_prompt.format(target=state["target_language"])},
                {"role": "user", "content": state["input"]}
            ]
        })
        
        translation = result["structured_response"]
        return {"deepseek_result": translation}
    except Exception as e:
        return {"deepseek_result": Translation(
            text="DeepSeek模型调用失败，请检查服务是否正常",
            input=state["input"],
            reason=str(e)
        )}

# 节点3：本地模型
def llm_translation_local(state: State) -> State:
    try:
        model = ChatOpenAI(
            model="Qwen3-235B", 
            api_key="EMPTY",  # 本地模型可能不需要API key
            base_url=local_base_url, 
            temperature=0.01
        )
        agent = create_agent(
            model=model,
            tools=[],
            response_format=ToolStrategy(Translation)
        )
        
        result = agent.invoke({
            "messages": [
                {"role": "system", "content": system_prompt.format(target=state["target_language"])},
                {"role": "user", "content": state["input"]}
            ]
        })
        
        translation = result["structured_response"]
        return {"local_result": translation}
    except Exception as e:
        return {"local_result": Translation(
            text="本地模型调用失败，请检查服务是否正常",
            input=state["input"],
            reason=str(e)
        )}

def summarize_translation(state: State) -> State:
    return {"summary": "翻译完成"}

# 构建 LangGraph 工作流
workflow = StateGraph(State)
workflow.add_node("llm_translation_qwen", llm_translation_qwen)
workflow.add_node("llm_translation_deepseek", llm_translation_deepseek)
workflow.add_node("llm_translation_local", llm_translation_local)
workflow.add_node("summarize_translation", summarize_translation)

workflow.add_edge(START, "llm_translation_qwen")
workflow.add_edge(START, "llm_translation_deepseek")
workflow.add_edge(START, "llm_translation_local")
workflow.add_edge("llm_translation_qwen", "summarize_translation")
workflow.add_edge("llm_translation_deepseek", "summarize_translation")
workflow.add_edge("llm_translation_local", "summarize_translation")
workflow.add_edge("summarize_translation", END)

# 编译工作流
graph = workflow.compile()

if __name__ == "__main__":
    # 测试用例
    input_text = "烧结机"
    state = {"input": input_text, "target_language": "英文"}
    result = graph.invoke(state)
    print(result)