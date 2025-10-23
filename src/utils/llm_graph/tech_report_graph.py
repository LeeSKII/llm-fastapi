from langgraph.graph import StateGraph, START, END
import httpx
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from typing_extensions import TypedDict
from langgraph.types import Send
import operator
from dotenv import load_dotenv
import logging

load_dotenv()

# 从环境变量获取API配置
tech_report_key = os.getenv("TECH_REPORT_KEY")
api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("QWEN_API_BASE_URL")
model_name = "qwen3-max"

# 定义状态类
class TechReportState(TypedDict):
    report_id: str
    report_year: str
    error: str
    source: dict
    intermediate_source: list
    partial_source: list
    subject: str
    intermediate_report: Annotated[list, operator.add]
    report: str

# 获取科技报告数据的函数
def get_report(state: TechReportState):
    """根据id获取科技简报数据"""
    logging.info(f"开始执行获取科技简报数据，报告ID: {state['report_id']}, 年份: {state['report_year']}")
    # 请求 URL
    url = "https://edb.cie-cn.com:8066/api/ext/attendanceMeal/getTechnologyBriefingWriting"
    # JSON 请求体
    json_data = {
        "key": tech_report_key,
        "year": state['report_year'],
        "id": state['report_id'],
    }

    # 发送 POST 请求
    response = httpx.post(
        url,
        json=json_data,  # 自动设置 Content-Type: application/json
        timeout=10.0  # 可选：设置超时
    )
    
    logging.info(f"请求科技简报数据，请求地址: {url}, 请求参数: {json_data}, 状态码: {response.status_code}, 响应内容: {response.json()}")

    # 处理响应
    if response.status_code == 200: 
        report_data = response.json()
        if report_data and 'data' in report_data:
            return {"source": report_data['data']}
    else:
        error_msg = f"请求失败，状态码: {response.status_code}, 错误: {response.text}"
        return {"error": error_msg}

# 系统提示
system_message = """**角色 (Role):**
你是一位严谨细致的科技报告编辑。你的核心任务是将结构化的JSON数据转换为一份流畅、专业且完全忠实于原文的报告。

**任务 (Task):**
根据我提供的包含科技简报信息的JSON字符串，生成一份正式的报告。在执行任务时，你必须严格遵守以下所有规则。

**核心规则 (Core Rules):**

1.  **绝对忠实原文 (Absolute Fidelity to Source):**
    *   **禁止遗漏 (No Omissions):** JSON中提供的每一个知识点、数据、名称或细节都必须完整地出现在最终报告中。不允许跳过任何条目。
    *   **禁止篡改 (No Alteration):** 严禁修改、歪曲或推断任何原文中未明确提及的信息。你的工作不是解读或再创作，而是忠实地呈现。

2.  **有限的润色 (Limited Polishing):**
    *   你的唯一编辑权限是进行基础的语言润色，以增强报告的可读性和专业性。
    *   这包括：修正语法错误、调整语序使句子更通顺、使用连接词使上下文更连贯。
    *   润色的目标是把零散的要点（bullet points）转化为通顺的段落，但绝不能改变其原始含义。

3.  **内容聚合 (Content Aggregation):**
    *   所有隶属于同一个 段落标题 条目的内容（通常在 `cont` 字段中），必须被整合在同一个段落或章节下。在完成当前条目的所有内容之前，不得开始下一个条目。

**输入数据说明**

`firdl1`表示一级段落标题，`firdl2`表示二级段落标题，`thierddl`表示三级段落标题。

`xh`表示序号。

**整理说明**

- 按照段落分级组织
- 相同的段落必须合并到同一层级
- 内容按照顺序从小到大排列
- 必须严格按照段落进行排版，如果不存在`firdl2`二级段落，`thierddl`三级段落，则不设置段落标题，严禁虚拟任何段落标题
- 内容必须按顺序进行编号，每一段内容结尾应该是分号，每一小节的最后一个内容结尾为句号

**输出格式说明**

# [subject 主题，通常为一级标题]

## sub-subject 子主题，通常为二级标题(如果存在 `firdl2` 字段才显示此标题)

### sub-sub-subject 子子主题，通常为三级标题(如果存在 `thierddl` 字段才显示此标题)

内容需要按顺序进行排版，不能跳过和省略任何条目。

例如：
1. 第一条内容；
2. 第二条内容。

**指令 (Instruction):**
现在，请根据上述所有规则和格式要求，处理用户提供的数据，并生成最终报告，生成的最终报告必须是纯净的Markdown文档，严格禁止外部嵌套```markdown```标签。"""

# 节点函数
def split_node(state: TechReportState) -> TechReportState:
    """按照一级段落合并数据"""
    result_dic = {}
    # 按照一级段落合并
    for item in state['source']:
        if item['firdl1'] not in result_dic:
            result_dic[item['firdl1']] = []
            result_dic[item['firdl1']].append(item)
        else:
            result_dic[item['firdl1']].append(item)
    # 内部顺序按照序号排序
    for key in result_dic.keys():
        result_dic[key] = sorted(result_dic[key], key=lambda x: x['xh'])
    return {"intermediate_source": result_dic}

def dispatch_node(state: TechReportState) -> TechReportState:
    """分发节点，提取主题列表"""
    return {"subject": list(state['intermediate_source'].keys())}

def generate_report_node(state: TechReportState) -> TechReportState:
    """生成报告节点"""
    # 初始化LLM
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.01
    )
    
    llm_response = llm.invoke([
        {"role": "system", "content": system_message},
        {"role": 'user', "content": f"当前主题是{state['subject']},提供的内容：{state['partial_source']}"}
    ])

    partial_report = [{'subject': state['subject'], 'report': llm_response.content}]
    return {"intermediate_report": partial_report}

def merge_node(state: TechReportState) -> TechReportState:
    """合并报告节点"""
    report = ""
    subject_index_dict = {}
    # 根据来源提取出最小一级段落的序号
    for key in state['intermediate_source'].keys():
        subject_index_dict[key] = state['intermediate_source'][key][0]['xh']
    # 一级段落根据序号从小到大排序
    sorted_keys = sorted(subject_index_dict, key=lambda k: subject_index_dict[k])
    # 按一级段落取出中间报告进行合并
    for key in sorted_keys:
        report += list(filter(lambda x: x['subject'] == key, state['intermediate_report']))[0]['report'] + "\n\n"
    return {"report": report}

# 条件路由函数
def route_to_generate(state: TechReportState):
    """路由到生成节点"""
    keys = state['subject']
    return [Send("generate_report_node", {"subject": key, "partial_source": state['intermediate_source'][key]}) for key in keys]

# 构建工作流
workflow = StateGraph(TechReportState)
workflow.add_node("get_report_data",get_report)
workflow.add_node("split_node", split_node)
workflow.add_node("dispatch_node", dispatch_node)
workflow.add_node("generate_report_node", generate_report_node)
workflow.add_node("merge_node", merge_node)

workflow.add_edge(START, "get_report_data")
workflow.add_edge("get_report_data", "split_node")
workflow.add_edge("split_node", "dispatch_node")
workflow.add_conditional_edges("dispatch_node", route_to_generate)
workflow.add_edge("generate_report_node", "merge_node")
workflow.add_edge("merge_node", END)

# 编译工作流
graph = workflow.compile()

if __name__ == "__main__":
    # 测试用例
    state = {"report_id": "c0ef14134cf34e5cbb0a21edfd9cb37d", "report_year": "2025"}
    logging.info("开始测试")
    result = graph.invoke(state)
    print(result['report'])