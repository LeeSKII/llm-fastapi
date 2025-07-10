import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("SELF_HOST_URL")
model_name = "qwen-plus-latest"
model_name = "Qwen3-235B"

# Initialize the LLM
llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    temperature=0.01
)

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
{format_instructions}
{query}
"""

standard_check_prompt = """no_think,你是一个专业的标准检测分析助手，任务是识别提供的国家标准（GB）和行业标准（如JB、QB等）是否过期以及有效，根据提供的标准号、标准名称，并验证标准名称与标准号是否匹配。以下是具体要求：
输入内容：接收包含标准号（如GB 12345-2018、JB/T 7890-2015等）、标准名称以及可能的发布日期文本。
识别过期标准：
检查标准号，结合你所知道的标准状态或有效性信息，判断标准是否过期。
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
{format_instructions}
{query}
"""

# Initialize Parsers
extract_parser = PydanticOutputParser(pydantic_object=StandardList)
check_parser = PydanticOutputParser(pydantic_object=StandardCheck)

# Define PromptTemplates
extract_prompt = PromptTemplate(
    template=standard_prompt,
    input_variables=["query"],
    partial_variables={"format_instructions": extract_parser.get_format_instructions()},
)

check_prompt = PromptTemplate(
    template=standard_check_prompt,
    input_variables=["query"],
    partial_variables={"format_instructions": check_parser.get_format_instructions()},
)

# Define Async Node Functions
async def extract_standards(state: GraphState) -> GraphState:
    """Extract standards from input text asynchronously."""
    input_text = state["input_text"]
    chain = extract_prompt | llm | extract_parser
    result = await chain.ainvoke({"query": input_text})
    return {"extracted_standards": result, "check_results": state.get("check_results", [])}

async def check_single_standard(standard, chain, semaphore):
    """Check a single standard with semaphore to limit concurrency."""
    async with semaphore:
        query = f"name:{standard.name},code:{standard.code}"
        result = await chain.ainvoke({"query": query})
        return result

async def check_standards_async(standards, chain):
    """Process standards concurrently with a maximum concurrency of 5."""
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent tasks
    tasks = [
        check_single_standard(standard, chain, semaphore)
        for standard in standards
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if not isinstance(result, Exception)]

async def check_standards(state: GraphState) -> GraphState:
    """Check each extracted standard for validity and matching with max concurrency of 5."""
    standards = state["extracted_standards"].standards
    chain = check_prompt | llm | check_parser
    check_results = await check_standards_async(standards, chain)
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