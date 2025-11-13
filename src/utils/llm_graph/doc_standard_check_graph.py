import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.tools import tool
from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("SELF_HOST_URL")
# model_name = "qwen-plus-latest"
model_name = "Qwen3-235B"
# Initialize Tavily client
tavily_api_key = os.getenv('TAVILY_API_KEY')
tavily_client = TavilyClient(api_key=tavily_api_key)

# Define Pydantic Models
class Standard(BaseModel):
    name: str = Field(description="标准|法规名称")
    code: str = Field(description="标准|法规 编号|日期")

class StandardList(BaseModel):
    standards: List[Standard] = Field(description="List of standards")

class StatusEnum(str, Enum):
    VALID = "有效"
    INVALID = "废止"
    PENDING = "待实施"
    UNKNOWN = "无法判断"

class StandardCheck(BaseModel):
    name: str = Field(description="标准|法规名称")
    code: str = Field(description="标准|法规 编号|日期")
    status: StatusEnum = Field(description="状态，枚举值：有效、废止、待实施、无法判断")
    isMatch: bool = Field(description="标准号与名称是否匹配")
    reason: str = Field(description="原因")

# Define the Graph State
class GraphState(TypedDict):
    input_text: str
    extracted_standards: StandardList
    check_results: List[StandardCheck]

# Define Prompts
standard_prompt = """no_think,# 提取标准名称和标准号的提示词

## 任务描述
你是一位专业的文档分析师，负责从用户提供的中文文章或文本中提取标准名称和标准号，并以给定的json schema格式输出。目标是准确识别所有提到的标准（包括国家标准、法规、行业标准等），提取其名称（不含引号）和编号（或发布/施行日期），并确保输出清晰、格式统一，适配structure_output模式。

## 规则
请按照以下步骤处理用户提供的中文文本，并生成结构化输出：

1. **输入接收**：接收用户提供的中文文章或文本片段。
2. **提取过程**：
   - 扫描文本，识别所有提到的国家标准、行业标准、规范、法律或法规。
   - 提取每个标准的名称（去除原文中的引号，如《》）和标准号（如 GB50406-2007）或发布/施行日期（如 2015年1月1日起施行）。
   - 如果标准号和日期同时存在，优先提取标准号；如果仅有日期，则记录日期。
   - 忽略非标准相关的文本，确保只提取明确的标准信息。
3. **输出格式**：
   - 单个标准Standard对象包含的字段如下：
     - **标准名称**：标准的完整名称，不含引号（如 中华人民共和国环境保护法）。
     - **标准号/日期**：标准编号（如 GB50406-2007）或发布/施行日期（如 2015年1月1日起施行）。如果没有编号，留空，示例:""。
   - 如果文本中没有标准信息，返回对象的standards字段为空列表[]
   - 最终输出的StandardList包含了一个standards字段，数据类型为List[Standard]。
   - 输出适配structure_output模式，确保表格格式清晰，字段规范。
   - Finally output, provide a list of standards in JSON format, wrapped in a 'standards' field as specified in the schema.
4. **约束**：
   - 仅提取明确的标准化文件名称和编号/日期，排除非标准化的描述。
   - 标准名称不含原文中的引号（如《》），直接提取核心名称。
   - 如果标准号或日期格式不一致（如括号内日期或无括号），统一提取并记录原文格式。
   - 所有输出均使用中文，表格格式清晰，易于阅读。
   - 如果标准名称或编号不完整（如缺少编号），编号字段留空：""。
5. 仔细审查文档，严格禁止遗漏任何可能出现的标准信息。
"""

standard_check_prompt = """no_think,你是一个专业的标准检测分析助手，任务是识别提供的国家标准（GB）和行业标准（如JB、QB等）是否过期以及有效，根据提供的标准号、标准名称，并验证标准名称与标准号是否匹配。
你可以使用提供的Tavily搜索工具来查询标准的有效性状态，优先使用搜索结果中的'answer'字段提供的信息。
以下是具体要求：
输入内容：接收包含标准号（如GB 12345-2018、JB/T 7890-2015等）、标准名称以及可能的发布日期文本。
识别过期标准：
检查标准号，优先使用提供的websearch结果进行判断，重点关注其中answer字段。
如果websearch结果无法判断标准有效性，结合你所知道的标准状态或有效性信息，判断标准是否有效。
识别标准号和名称：
识别标准号（如GB 12345-2018、JB/T 7890-2015）。
识别标准名称（如《家用电器安全标准》、《机械设备技术规范》）。
确保标准号与名称一一对应，避免混淆。
验证名称与标号匹配：
检查标准号与标准名称是否逻辑一致（例如，GB 12345-2018是否对应《家用电器安全标准》）。
如果发现标准号与名称不匹配（如GB 12345-2018对应《食品卫生标准》），标记为不匹配并说明可能的原因（如录入错误或标准号更新）。
你每次都会接收到一对标准号和名称，需要对其进行验证。
输出格式：
以清晰的结构化形式输出结果，包含以下内容：
标准号
标准名称
状态（有效/废止/待实施/无法判断）
名称与标号是否匹配（是/否）
原因（判断的依据和理由）
"""

# Define PromptTemplates
extract_prompt = PromptTemplate(
    template=standard_prompt,
    input_variables=["query"],
)

check_prompt = PromptTemplate(
    template=standard_check_prompt,
    input_variables=["query"],
)

async def tavily_search(query: str):
    """Search for information about a standard's validity status using Tavily."""
    # 将同步的 tavily_client.search 运行在单独的线程中以避免阻塞
    response = await asyncio.to_thread(
        tavily_client.search,
        query,
        search_depth='advanced',
        include_answer=True
    )
    return {
        "answer": response.get('answer', ''),
        "results": response.get('results', [])
    }

# Define Async Node Functions
def extract_standards(state: GraphState) -> GraphState:
    """Extract standards from input text asynchronously."""
    input_text = state["input_text"]
    # Initialize the LLM
    model = ChatOpenAI(model=model_name,api_key=api_key,base_url=base_url,temperature=0)
    agent = create_agent(
        model=model,
        tools=[],
        response_format=ToolStrategy(StandardList)
    )

    result = agent.invoke({"messages":[{"role":"system","content":standard_prompt},{"role":"user","content":input_text}]})
    extracted_standards = result["structured_response"]
    return {"extracted_standards": extracted_standards, "check_results": state.get("check_results", [])}

async def check_single_standard(standard, agent, semaphore):
    """Check a single standard with semaphore to limit concurrency."""
    async with semaphore:
        query = f"name:{standard.name},code:{standard.code}"
        web_search_result = await tavily_search(f"标准号:{standard.code} 当前是否有效")
        model = ChatOpenAI(model=model_name,api_key=api_key,base_url=base_url,temperature=0)
        agent = create_agent(
        model=model,
        tools=[],
        response_format=ToolStrategy(StandardCheck)
    )
        result = agent.invoke({"messages":[{"role":"system","content":standard_check_prompt},{"role":"user","content":f"查询到的相关资料：{web_search_result}，当前待检测标准：{query}，检查和判断是否有效？"}]})
        return result

async def check_standards_async(standards, agent):
    """Process standards concurrently with a maximum concurrency of 5."""
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent tasks
    tasks = [
        check_single_standard(standard, agent, semaphore)
        for standard in standards
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if not isinstance(result, Exception)]

async def check_standards(state: GraphState) -> GraphState:
    """Check each extracted standard for validity and matching with max concurrency of 5."""
    standards = state["extracted_standards"].standards
    model = ChatOpenAI(model=model_name,api_key=api_key,base_url=base_url,temperature=0)
    agent = create_agent(
        model=model,
        tools=[],
        response_format=ToolStrategy(StandardList)
    )
    check_results = await check_standards_async(standards, agent)
    return {"check_results": check_results}

# Define the Workflow
workflow = StateGraph(GraphState)

# Add async nodes
workflow.add_node("extract_standards", extract_standards)
workflow.add_node("check_standards", check_standards)

# Define edges
workflow.add_edge("extract_standards", "check_standards")
workflow.add_edge("check_standards", END)

# Set entry point
workflow.set_entry_point("extract_standards")

# Compile the graph for async execution
graph = workflow.compile()

# 异步调用
async def main():
    state = {'input_text':"5）施布置合理、操作安全、简便，尽量减小项目实施时对现有单元生产的影响；6）严格执行国家、地方及企业的有关环保、安全卫生、节能、工程设计统一技术规定等有关标准、规范。7）执行的设计规范：《中华人民共和国环境保护法》（2015年1月1日起施行）《建设项目环境保护管理条例》（1998年11月29日发布施行）《钢铁工业环境保护设计规范》（GB50406-2007）《钢铁烧结、球团工业大气污染物排放标准》（GB28662-2012）《建筑地基基础设计规范》（GB50007-2011）《建筑结构荷载规范》 （GB 50009-2012）《混凝土结构设计规范》（GB50010-2010）《钢结构设计规范》（GB50017-2003）《砌体结构设计规范》（GB50003-2011）。《建筑抗震设计规范》（GB50011-2010）。《建筑工程抗震设防分类标准》（GB50223－2008）。《动力机器基础设计规范》（GB50040-96）。《烟囱设计规范》（GB50051-2013）。《建筑桩基技术规范》（JGJ94-2008）。《岩土工程勘察规范》GB 50021-2001 （2009年版）。《通用用电设备配电设计规范》   GB50055-2011《建筑物防雷设计规范》     GB50057-2010《建筑设计防火规范》       GB50016-2014 《3～110KV高压配电装置设计规范》   GB50060-2008《供配电系统设计规范》     GB50052-2009《低压配电设计规范》       GB50054-2011《20kV及以下变电所设计规程》GB50053-2013"}
    result = await graph.ainvoke(state)
    print(result['check_results'])

if __name__ == '__main__':
    asyncio.run(main())