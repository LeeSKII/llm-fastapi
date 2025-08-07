from langgraph.graph import StateGraph, START, END
from typing import TypedDict,Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage
import os
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

qwen_api_key = os.getenv("QWEN_API_KEY")
qwen_base_url = os.getenv("QWEN_API_BASE_URL")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_base_url = os.getenv("DEEPSEEK_API_BASE_URL")
local_base_url = os.getenv("SELF_HOST_URL")

class Translation(BaseModel):
    text: str = Field(...,description="翻译后的文本")
    input: str = Field(...,description="输入文本")
    reason: str = Field(...,description="翻译解释")

parser = PydanticOutputParser(pydantic_object=Translation)
format_instructions = parser.get_format_instructions()
system_prompt = """你是一位专业且严谨的工程设计领域翻译专家，擅长将用户输入的任何目标语言翻译成目标语言{target}，请严格遵循输出的json格式{format_instructions}，按要求翻译每轮对话用户提供的输入。"""
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
      llm = ChatOpenAI(model="qwen-plus-latest",api_key=qwen_api_key,base_url=qwen_base_url,temperature=0.01)
      llm_with_structured_output = llm.with_structured_output(Translation).with_retry(stop_after_attempt=3)
      response = llm_with_structured_output.invoke([SystemMessage(system_prompt.format(target=state["target_language"],format_instructions=format_instructions)), HumanMessage(state["input"])])
      return {"qwen_result": response}
    except Exception as e:
        return {"qwen_result": Translation(text="千问模型调用失败，请检查服务是否正常",input=state["input"],reason=str(e))}

# 节点2：Deepseek
def llm_translation_deepseek(state: State) -> State:
    try:
      llm = ChatOpenAI(model="deepseek-chat",api_key=deepseek_api_key,base_url=deepseek_base_url,temperature=0.01)
      llm_with_structured_output = llm.with_structured_output(Translation,method="json_mode").with_retry(stop_after_attempt=3)
      response = llm_with_structured_output.invoke([SystemMessage(system_prompt.format(target=state["target_language"],format_instructions=format_instructions)), HumanMessage(state["input"])])
      return {"deepseek_result": response}
    except Exception as e:
        return {"deepseek_result": Translation(text="DeepSeek模型调用失败，请检查服务是否正常",input=state["input"],reason=str(e))}

# 节点3：本地模型
def llm_translation_local(state: State) -> State:
    try:
      # 本地模型调用略有不同
      client: OpenAI = OpenAI(
          api_key="EMPTY",
          base_url=local_base_url,
      )
      completion = client.chat.completions.create(
          model="Qwen3-235B",
          messages=[
              {
                  "role": "system",
                  "content": system_prompt.format(target=state["target_language"],format_instructions=format_instructions),
              },
              {
                  "role": "user",
                  "content": state["input"],
              }
          ],
          extra_body={"guided_json": Translation.model_json_schema()},
      )
      response = completion.choices[0].message.reasoning_content
      translation = Translation.model_validate_json(response)
      return {"local_result": translation}
    except Exception as e:
        return {"local_result": Translation(text="本地模型调用失败，请检查服务是否正常",input=state["input"],reason=str(e))}

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