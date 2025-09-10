import datetime
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.config import get_stream_writer
from typing import TypedDict,Annotated,List,Optional,Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage
import os
from pydantic import BaseModel,Field
from langchain.output_parsers import PydanticOutputParser
from openai import OpenAI
import lancedb
import pandas as pd
import logging
from textwrap import dedent
import operator

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.utils import logger

from dotenv import load_dotenv
load_dotenv()

qwen_api_key = os.getenv("QWEN_API_KEY")
qwen_base_url = os.getenv("QWEN_API_BASE_URL")
contract_db_path = os.getenv("CONTRACT_DB_PATH")
MODEL_NAME = "qwen-plus"

def get_embedding(text,model='text-embedding-v4',dimensions=2048):
    client = OpenAI(
        api_key=qwen_api_key, 
        base_url=qwen_base_url
    )

    completion = client.embeddings.create(
        model=model,
        input=text,
        dimensions=dimensions, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
        encoding_format="float"
    )
    
    return completion.data[0].embedding

def custom_check_point_output(data: dict):
    """
    自定义检查点输出函数
    
    Args:
        data (dict): 要输出的数据
    """
    writer = get_stream_writer()  
    writer(data) 

class SearchKeyWords(BaseModel):
    contract_meta_info_key_words:Optional[List[str]]=Field(None,description="合同元数据查询关键词")
    equipments_key_words:Optional[List[str]]=Field(None,description="合同设备查询关键词")

class AdvancedSearchConditions(BaseModel):
    year_condition: Optional[str] = Field(None,description="年份查询范围条件，枚举值，只能在['=','<','>']三个中选择其中一个或者为None，None表示不进行年份条件过滤，'='表示等于某年份，'<'表示小于某年份，'>'表示大于某年份")
    year: Optional[str] = Field(None,description="指定查询的年份")
    equipment: Optional[str] = Field(None,description="查询设备的名称关键字")
    equipment_condition: Optional[str] = Field(None,description="设备条件，枚举值，只能在['all','partial']两个中选择其中一个或者为None，None表示不进行设备查询，'all'表示需要查询所有设备，'partial'表示查询部分设备")

    
# 定义 LangGraph 状态
class State(TypedDict):
    query: str
    meta_search_query_keyword: SearchKeyWords
    vector_search_contract:pd.DataFrame
    keyword_search_contract:pd.DataFrame
    filtered_contracts: List[Any]
    contract_info_checked:Annotated[list[dict], operator.add]
    need_year_condition: bool # 是否需要按照年份条件过滤
    need_equipment_condition: bool # 是否需要按照设备条件过滤
    advanced_filter_conditions: AdvancedSearchConditions # 高级过滤条件
    contract_filtered_final: Annotated[list[dict], operator.add] # 最终过滤后的合同列表
    messages:List[dict]
    response:str
    error:str

def generate_search_words(state: State) -> State:
    try:
      logging.info(f"generate_search_words,开始进行合同处理：收到用户请求数据：{state}")
      llm = ChatOpenAI(model=MODEL_NAME,api_key=qwen_api_key,base_url=qwen_base_url,temperature=0.01)
      llm_with_structured_output = llm.with_structured_output(SearchKeyWords).with_retry(stop_after_attempt=3)
      parser = PydanticOutputParser(pydantic_object=SearchKeyWords)
      format_instructions = parser.get_format_instructions()
      system_prompt = dedent(f"""你是一个合同查询助手，负责拆解用户查询需求中的关键字。关键字分为两类：
                            1.  **合同元信息关键字**：如项目名称、供应商、工程子项等非设备信息。
                            2.  **设备关键字**：如风机、锅炉、泵等设备信息。
                            # 核心规则
                            **绝对禁止虚构用户未提供的任何关键字。你只能严格地从用户输入的消息中提取关键词。**
                            # 查询分析步骤
                            1.  **分析合同元信息关键字**：检查用户查询是否包含项目、供应商、工程等非设备信息。如有，则提取相应名称作为关键字。
                            2.  **分析设备关键字**：检查用户查询是否包含设备信息。如有，则提取设备名称作为关键字。
                            ## 非设备类查询分析步骤：
                            1.检查用户的查询是否包含非设备类查询需求；
                            2.识别用户的查询需求属于合同的元数据关键字，例如项目名称、供应商、工程子项等非设备相关信息的查询关键字。
                            ## 设备类查询分析步骤：
                            1.检查用户的查询是否包含设备类查询需求；
                            2.识别用户的查询需求属于设备相关信息的设备关键字。
                            # 关键字提取规范
                            -   **保持简洁**：提取的核心名称，去除“项目”、“合同”、“采购”等泛用后缀词。例如，“湖南华菱涟钢项目”应提取为“湖南华菱涟钢”。
                            -   **忽略通用词汇**：如“有哪些”、“合同”、“查询”等不应被提取为关键字。
                            # 示例：
                            1.用户提问：湖南华菱涟钢项目和揭阳大南海石化工业区危险废物焚烧以及宝山钢铁四烧结余热锅炉价格对比
                            第一步执行非设备类查询分析步骤分析：首先判断这是一个非设备类查询，先提取属于合同的元数据关键字：湖南华菱涟钢，揭阳大南海石化工业区危险废物焚烧、宝山钢铁四烧结分别是对应的项目名称，项目名称属于合同元数据，所以提取的非设备类关键词有湖南华菱涟钢、揭阳大南海石化工业区危险废物焚烧、宝山钢铁四烧结，这里的精简提取注意事项是：例如「湖南华菱涟钢项目」，关键字为「湖南华菱涟钢」，不是「湖南华菱涟钢项目」，关键字不需要带额外的「项目」两个字；
                            第二步执行设备类查询分析步骤：用户的查询中有余热锅炉，这属于工程设备，所以识别用户的查询这设备关键字：余热锅炉。
                            返回结果：contract_meta_info_key_words为[湖南华菱涟钢,揭阳大南海石化工业区危险废物焚烧,宝山钢铁四烧结]，equipments_key_words为[余热锅炉]。
                            2.用户提问：增压风机的采购合同有哪些
                            第一步执行非设备类查询分析步骤分析：首先检查用户的提问只询问了采购合同，没有进一步具体的合同元信息，因此不属于非设备类查询，无查询关键字；
                            第二步执行设备类查询分析步骤：用户提到了增压风机，这是一种设备，所以这是一个设备类查询，提取设备关键字：增压风机。
                            返回结果：contract_meta_info_key_words为null，equipments_key_words为[增压风机]。
                            3.用户提问：宝钢德盛项目有多少合同
                            第一步执行非设备类查询分析步骤分析：用户的提到了宝钢德盛项目，项目属于合同元信息，因此判断为非设备类查询，查询关键字：[宝钢德盛]；
                            第二步执行设备类查询分析步骤：用户提问没有涉及到设备信息，因此不属于设备类查询，提取设备关键字：null。
                            返回结果：contract_meta_info_key_words为[宝钢德盛]，equipments_key_words为null。
                            4.用户提问：金通灵科技有哪些合同
                            第一步执行非设备类查询分析步骤分析：用户的提到了金通灵科技，这属于某个公司的名称，可能属于供应商，用户可能是想要查询该供应商涉及到的合同，供应商属于合同元信息，因此判断为非设备类查询，查询关键字：[金通灵科技]；
                            第二步执行设备类查询分析步骤：用户提问没有涉及到设备信息，因此不属于设备类查询，提取设备关键字：null。
                            返回结果：contract_meta_info_key_words为[金通灵科技]，equipments_key_words为null。
                            需要按照提供的json格式进行输出：\n{format_instructions}\n""")
      logging.info(f"generate_search_words,生成合同处理系统提示：{system_prompt}")
      response = llm_with_structured_output.invoke([SystemMessage(system_prompt),*state["messages"], HumanMessage(state["query"])])
      logging.info(f"generate_search_words,生成查询关键字格式化输出：{response}")
      custom_check_point_output({'type':'update_info','node':'generate_search_words','data':response})
      return {"meta_search_query_keyword": response}
    except Exception as e:
        return {"meta_search_query_keyword": SearchKeyWords(contract_meta_info_key_words=None,equipments_key_words=None),"error":"generate_search_words.\n"+str(e)}
    
def vector_search(state: State)->State:
    try:
       logging.info(f"vector_search,开始进行向量化查询：收到用户请求数据：{state}")
       if state["meta_search_query_keyword"].contract_meta_info_key_words is not None:
           df_list = []
           db = lancedb.connect(contract_db_path) 
           table = db.open_table("contract_table")
           for word in state["meta_search_query_keyword"].contract_meta_info_key_words:
               embedding = get_embedding(word)
               search_vector_results = table.search(embedding,vector_column_name="meta_vector").limit(10).to_pandas()
               df_list.append(search_vector_results)
           search_results = pd.concat(df_list, ignore_index=True)
           # 根据组合列去重（保留第一个出现的值）
           final_df = search_results.drop_duplicates(subset=['contact_no', 'project_name'])
           logging.info(f"vector_search,向量化查询结果：{final_df[['contact_no', 'project_name']].to_dict(orient='records')}")
           return {"vector_search_contract": final_df}
       else:
            return {"vector_search_contract": None}
    except Exception as e:
        return {"vector_search_contract": None,"error":"vector_search.\n"+str(e)}
    
def keyword_search(state: State)->State:
    try:
        logging.info(f"keyword_search,开始进行关键字查询：收到用户请求数据：{state}")
        df_list = []
        db = lancedb.connect(contract_db_path) 
        table = db.open_table("contract_table")
        merged_list = (state["meta_search_query_keyword"].contract_meta_info_key_words or []) + (state["meta_search_query_keyword"].equipments_key_words or [])
        
        for word in merged_list:
            logging.info(f"keyword_search,开始进行关键字查询：查询语句：doc LIKE '%{word}%'")
            search_like_results = table.search().where(f"doc LIKE '%{word}%'").limit(20).to_pandas()
            logging.info(f"keyword_search,开始进行关键字查询：查询结果：{len(search_like_results)}")
            df_list.append(search_like_results)

        search_results = pd.concat(df_list, ignore_index=True)
        # 根据组合列去重（保留第一个出现的值）
        final_df = search_results.drop_duplicates(subset=['contact_no', 'project_name'])
        logging.info(f"keyword_search,关键字查询结果：{final_df[['contact_no', 'project_name']].to_dict(orient='records')}")   
        return {"keyword_search_contract": final_df}
    except Exception as e:
        return {"keyword_search_contract": None,"error":"keyword_search.\n"+str(e)}

def filter_contracts(state: State)->State:
    '''进行合同的过滤处理'''
    try:
        # 合并两个来源的df，排除重复数据
        final_df = pd.concat([state["keyword_search_contract"],state["vector_search_contract"]], ignore_index=True)
        logging.info(f"filter_contracts,合并两项查询，条目数：{len(final_df)}")
        final_df = final_df.drop_duplicates(subset=['contact_no', 'project_name'])
        logging.info(f"filter_contracts,取消重复数据，条目数：{len(final_df)}")
        return {"filtered_contracts": final_df.to_dict(orient="records")}
    except Exception as e:
        return {"filtered_contracts": [],"error":"filter_contracts.\n"+str(e)}
    
def continue_check_contract_belong(state: State):
    '''检查合同是否属于用户查询范围'''
    return [Send("check_contract_belong", {"query": state["query"], "contract_meta": s["contract_meta"],"date":s['date'], "equipment_table": s["equipment_table"]}) for s in state['filtered_contracts']]

def check_contract_belong(state: dict)->State:
    '''检查合同是否属于用户查询范围'''
    query = state["query"]
    contract_info = state['contract_meta']+state['equipment_table']
    try:
        system_prompt = f"你是一个精确的合同查询判断机器，请根据用户提供的合同信息和用户提出的问题，判断合同是否属于用户查询范围，输出false表示不属于，true表示属于，记住，只需要输出true或false，不要输出其它任何信息。"
        llm = ChatOpenAI(model=MODEL_NAME,api_key=qwen_api_key,base_url=qwen_base_url,temperature=0.01)
        response = llm.invoke([SystemMessage(system_prompt), HumanMessage(f"用户提供的合同信息：{contract_info},\n用户提出的问题：{query}")])
        if "true" in response.content.lower():
            return {"contract_info_checked": [{"contract_meta":state['contract_meta'],"date":state['date'],"equipment_table":state['equipment_table']}]}
        else:
            return {"contract_info_checked": []}
    except Exception as e:
        return {"error":"check_contract_belong.\n"+str(e)}

def advanced_filter_contracts(state: State)->State:
    '''进行合同的额外过滤处理，例如查询指定的某种设备，就提取相关设备的数据，查询指定年份的就限定相关年份的数据'''
    try:
        query = state["query"]
        contract_list = state["contract_info_checked"]
        logging.info(f"advanced_filter_contracts,参与合同高级过滤的合同数据：{contract_list}")
        llm = ChatOpenAI(model=MODEL_NAME,api_key=qwen_api_key,base_url=qwen_base_url,temperature=0.01)
        llm_with_structured_output = llm.with_structured_output(AdvancedSearchConditions).with_retry(stop_after_attempt=3)
        parser = PydanticOutputParser(pydantic_object=AdvancedSearchConditions)
        format_instructions = parser.get_format_instructions()
        system_prompt = dedent(f"""你是一位经验丰富的合同分析师，根据用户提出的问题，分析是否需要进一步执行合同筛选。
                               执行进一步筛选的条件类型目前有两种：
                               1.按年份筛选：用户明确要求需要执行的年份范围的筛选
                               示例1：“2021年以前的合同”,year=[2021],year_condition="="
                               示例2：“2022年以后的合同”，year=[2022],year_condition=">"
                               示例3：“近5年的合同”，假如现在是2025年，往前推5年，year=[2020],year_condition=">"
                               示例4：未包含年份条件，year=None,year_condition=None
                               2.按设备筛选：用户可以指定查询的设备关键字，已经提供的设备关键字有：{state['meta_search_query_keyword'].equipments_key_words}。
                               示例1：假如已经提供的设备关键字有：风机、余热锅炉，equipment=[风机、余热锅炉]，equipment_condition="partial"
                               示例2：假如未提供设备关键字，但是用户的提问没有明确说不需要查询设备信息，那么需要查询全部设备：equipment=None,equipment_condition="all"
                               示例3：假如未提供设备关键字，但是用户的提问明确说只需要指定信息，例如供应商、合同金额等，那么不需要查询设备信息：equipment=None,equipment_condition=None
                               请根据以下条件进行筛选，按照提供的json格式进行输出：\n{format_instructions}\n当前时间是：{datetime.datetime.now().strftime('%Y-%m-%d')}。""")
        response:AdvancedSearchConditions = llm_with_structured_output.invoke([SystemMessage(system_prompt),*state["messages"], HumanMessage(query)])
        logging.info(f"advanced_filter_contracts,高级过滤条件：{response}")
        return {"need_equipment_condition": False if response.equipment_condition is None else True,"need_year_condition": False if response.year is None else True,"advanced_filter_conditions":response}
    except Exception as e:
        return {"error":"advanced_filter_contracts.\n"+str(e)}

def advanced_year_filter(state: State)->State:
    contract_list = state["contract_info_checked"]
    logging.info(f"advanced_year_filter,最终参与合同高级过滤的合同数据量：{len(contract_list)}")
    if state["advanced_filter_conditions"].year is False:
        return {}
    else:
        year_condition = state["advanced_filter_conditions"].year_condition
        if year_condition is not None:
            year = state["advanced_filter_conditions"].year
            condition = state["advanced_filter_conditions"].year_condition
            if condition == "=":
                contract_list = [c for c in contract_list if int(c['date'][:4]) == int(year)]
            elif condition == ">":
                contract_list = [c for c in contract_list if int(c['date'][:4]) > int(year)]
            elif condition == "<":
                contract_list = [c for c in contract_list if int(c['date'][:4]) < int(year)]
            logging.info(f"advanced_year_filter,通过{year_condition}参与合同高级过滤的合同数据量：{len(contract_list)}")
            logging.info(f"advanced_year_filter,参与合同高级过滤的合同数据：{contract_list}")
            return {"contract_info_checked": contract_list}
        else:
            return {}

def advanced_equipment_filter(state: State)->State:
    contract_list = state["contract_info_checked"]
    logging.info(f"advanced_equipment_filter,最终参与合同高级过滤的合同数据量：{len(contract_list)}")
    if state["need_equipment_condition"] is False:
        return {}
    else:
        return [Send("equipment_choice", {"query": state["query"],"condition":state["advanced_filter_conditions"], "contract_meta": s["contract_meta"],"date":s['date'], "equipment_table": s["equipment_table"]}) for s in contract_list]

def equipment_choice(state: dict)->State:
    query = state["query"]
    logging.info(f"equipment_choice,正在精确识别待查找的合同设备数据，输入数据：{state}")
    if state['condition'].equipment_condition == "all":
        return {"contract_filtered_final":[{"contract_meta":state['contract_meta'],"date":state['date'],"equipment_table":state['equipment_table']}]}
    elif state['condition'].equipment_condition is None:
        return {"contract_filtered_final":[{"contract_meta":state['contract_meta'],"date":state['date'],"equipment_table":None}]}
    else:      
        prompt = dedent(f"""你是一位合同设备分析机器人，你的任务是根据用户的提问，精确识别合同中所涉及的设备信息，包括可能涉及到的名称、规格型号、数量、价格等，并将其信息全部返回。
                            请注意，严格参考合同的设备信息，禁止遗漏提供的设备信息，严禁虚构任何不属于上下文的消息。
                            设备信息：\n{state['equipment_table']}
                            请根据以下规则回答用户提问：
                            1.如果提供的合同未包含用户询问的设备信息，则返回null;
                            2.如果合同中除了用户提问的设备还包含了其它设备，则只返回用户提问的设备信息；
                            3.如果合同中没有任何设备信息，则返回null；
                            4.如果合同中包含了用户提问的多个设备信息，则返回多个设备信息。
                            请根据以下合同信息回答用户提出的问题：\n{query}\n
                            回复规则：
                            你的回复只包含必要的结果信息，你是一台精确的执行机器，不需要提供任何其它无关的信息和格式例如```markdown```标记输出，使用Markdown格式回复。
                            """)
        llm = ChatOpenAI(model=MODEL_NAME,api_key=qwen_api_key,base_url=qwen_base_url,temperature=0.01,max_completion_tokens=8000)
        response = llm.invoke([{'role':'user', 'content': prompt}])
        logging.info(f"equipment_choice,精确识别的设备数据：{response.content}")
        return {"contract_filtered_final":[{"contract_meta":state['contract_meta'],"date":state['date'],"equipment_table":response.content}]}
    
def generate_response(state: State)->State:
    try:
        contract_list = state["contract_filtered_final"]
        logging.info(f"generate_response,最终参与合同筛选的合同数据：{contract_list}")
        custom_check_point_output({'type':'update_info','node':'generate_response','data':contract_list})
        system_prompt = dedent(f"""请根据以下合同信息回答用户提出的问题：\n{contract_list}\n
                               请注意，严格参考合同信息，尽量不要遗漏提供的合同信息，因为这些合同已经由上游检测程序校准过，确认属于用户的询问范围。
                               严禁虚构任何不属于上下文的消息:
                               1.如果提供的合同未覆盖用户到用户提问涉及的范围，例如查询某个项目或者设备的合同，提供的合同信息未找到该项目或者该设备，则必须显式声明该项目未查询到相关的合同信息或者未查询到相关设备合同信息;
                               当前时间是：{datetime.datetime.now().strftime('%Y-%m-%d')}。
                               不需要提供任何其它无关的信息和格式例如```markdown```标记输出，使用Markdown格式回复""")
        llm = ChatOpenAI(model=MODEL_NAME,api_key=qwen_api_key,base_url=qwen_base_url,temperature=0.01,max_completion_tokens=8000)
        response = llm.invoke([SystemMessage(system_prompt),*state["messages"], HumanMessage(state["query"])])
        logging.info(f"generate_response,最终回复：{response.content}")
        messages = [*state["messages"], {'role': 'user', 'content': state['query']},{'role': 'assistant', 'content': response.content}] 
        custom_check_point_output({'type':'final_response','node':'generate_response','data':{"response": response.content,"messages": messages}})
        return {"response": response.content,"messages": messages}
    except Exception as e:
        return {"response": None,"error":"generate_response.\n"+str(e)}

# 构建 LangGraph 工作流
workflow = StateGraph(State)
workflow.add_node("generate_search_words", generate_search_words)
workflow.add_node("vector_search", vector_search)
workflow.add_node("keyword_search", keyword_search)
workflow.add_node("filter_contracts", filter_contracts)
workflow.add_node("generate_response", generate_response)
workflow.add_node("check_contract_belong", check_contract_belong)
workflow.add_node("advanced_filter_contracts", advanced_filter_contracts)
workflow.add_node("advanced_year_filter", advanced_year_filter)
workflow.add_node("advanced_equipment_filter", advanced_equipment_filter)
workflow.add_node("equipment_choice",equipment_choice)

workflow.add_edge(START, "generate_search_words")
workflow.add_edge("generate_search_words", "vector_search")
workflow.add_edge("generate_search_words", "keyword_search")
workflow.add_edge("vector_search", "filter_contracts")
workflow.add_edge("keyword_search", "filter_contracts")
workflow.add_conditional_edges("filter_contracts", continue_check_contract_belong)
workflow.add_edge("check_contract_belong", "advanced_filter_contracts")
workflow.add_edge("advanced_filter_contracts", "advanced_year_filter")
workflow.add_conditional_edges("advanced_year_filter", advanced_equipment_filter)
workflow.add_edge("equipment_choice", "generate_response")
workflow.add_edge("generate_response", END)


# 编译工作流
graph = workflow.compile()

if __name__ == "__main__":
    # 测试用例
    query = "查询近三年的热电偶采购价格。"
    state = {"query": query,"messages":[]}
    result = graph.invoke(state)
    print(result)