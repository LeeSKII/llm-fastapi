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
    project_key_words:Optional[List[str]]=Field(None,description="项目关键词")
    equipments_key_words:Optional[List[str]]=Field(None,description="设备关键词")

# 定义 LangGraph 状态
class State(TypedDict):
    query: str
    meta_search_query_keyword: SearchKeyWords
    vector_search_contract:pd.DataFrame
    keyword_search_contract:pd.DataFrame
    filtered_contracts: List[Any]
    contract_info_checked:Annotated[list[dict], operator.add]
    messages:List[dict]
    response:str
    error:str

def generate_search_words(state: State) -> State:
    try:
      logging.info(f"generate_search_words,开始进行合同处理：收到用户请求数据：{state}")
      llm = ChatOpenAI(model="qwen-plus",api_key=qwen_api_key,base_url=qwen_base_url,temperature=0.01)
      llm_with_structured_output = llm.with_structured_output(SearchKeyWords).with_retry(stop_after_attempt=3)
      parser = PydanticOutputParser(pydantic_object=SearchKeyWords)
      format_instructions = parser.get_format_instructions()
      system_prompt = dedent("""你是一个合同查询的助手，用户提出的查询需求可能是关于某些项目的也可能是关于某些设备的，请将用户的查询分解成不同的查询关键字，你只能提取用户提供的查询关键字，这意味着你绝对禁止虚拟项目或者设备关键字，只能从用户的原始查询中提取，注意，你需要提取的是关键字，这个关键字用于执行SQL的精确搜索，所以一定要保持简洁，否则会导致严重的业务问题。
      这里有一些示例：
      1.用户提问：湖南华菱涟钢项目和揭阳大南海石化工业区危险废物焚烧以及宝山钢铁四烧结余热锅炉价格对比。
      分析：先提取项目关键字：湖南华菱涟钢，揭阳大南海石化工业区危险废物焚烧、宝山钢铁四烧结分别是对应的项目名称，注意这里的关键词是项目名称，所以关键词是湖南华菱涟钢，并不是湖南华菱涟钢项目，关键字不需要带额外的「项目」两个字；
      第二步检查设备关键字，这里的余热锅炉是某种工程设备，所以这里是一个设备关键字：余热锅炉。
      2.用户提问：增压风机的采购合同有哪些
      分析：先提取项目关键字：这里不涉及项目名称，所以不用提取；
      第二步检查设备关键字，这里的增压风机是一种设备，所以这里是一个设备关键字：增压风机。
      需要按照提供的json格式进行输出：\n{format_instructions}\n""")
      response = llm_with_structured_output.invoke([SystemMessage(system_prompt.format(format_instructions=format_instructions)),*state["messages"], HumanMessage(state["query"])])
      logging.info(f"generate_search_words,生成查询关键字格式化输出：{response}")
      custom_check_point_output({'type':'update_info','node':'generate_search_words','data':response})
      return {"meta_search_query_keyword": response}
    except Exception as e:
        return {"meta_search_query_keyword": SearchKeyWords(project_key_words=None,equipments_key_words=None),"error":"generate_search_words.\n"+str(e)}
    
def vector_search(state: State)->State:
    try:
       logging.info(f"vector_search,开始进行向量化查询：收到用户请求数据：{state}")
       if state["meta_search_query_keyword"].project_key_words is not None:
           df_list = []
           db = lancedb.connect(contract_db_path) 
           table = db.open_table("contract_table")
           for word in state["meta_search_query_keyword"].project_key_words:
               embedding = get_embedding(word)
               search_vector_results = table.search(embedding,vector_column_name="meta_vector").limit(5).to_pandas()
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
        db = lancedb.connect(r"C:\Lee\work\contract\csv\v3\contract_full_lancedb") 
        table = db.open_table("contract_table")
        merged_list = (state["meta_search_query_keyword"].project_key_words or []) + (state["meta_search_query_keyword"].equipments_key_words or [])
        
        for word in merged_list:
            logging.info(f"keyword_search,开始进行关键字查询：查询语句：doc LIKE '%{word}%'")
            search_like_results = table.search().where(f"doc LIKE '%{word}%'").limit(10).to_pandas()
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
    return [Send("check_contract_belong", {"query": state["query"], "contract_meta": s["contract_meta"], "equipment_table": s["equipment_table"]}) for s in state['filtered_contracts']]

# TODO: 关键字查询的结果不需要再次check，只需要检查向量化查询的数据
def check_contract_belong(state: dict)->State:
    '''检查合同是否属于用户查询范围'''
    query = state["query"]
    contract_info = state['contract_meta']+state['equipment_table']
    try:
        system_prompt = f"请根据用户提供的合同信息和用户提出的问题，判断合同是否属于用户查询范围，输出false表示不属于，true表示属于，记住，只需要输出true或false，不要输出其它任何信息。"
        llm = ChatOpenAI(model="qwen-plus",api_key=qwen_api_key,base_url=qwen_base_url,temperature=0.01)
        response = llm.invoke([SystemMessage(system_prompt), HumanMessage(f"用户提供的合同信息：{contract_info},\n用户提出的问题：{query}")])
        if "true" in response.content.lower():
            return {"contract_info_checked": [{"contract_meta":state['contract_meta'],"equipment_table":state['equipment_table']}]}
        else:
            return {"contract_info_checked": []}
    except Exception as e:
        return {"error":"check_contract_belong.\n"+str(e)}
    
def generate_response(state: State)->State:
    try:
        contract_list = state["contract_info_checked"]
        logging.info(f"generate_response,最终参与合同筛选的合同数据：{contract_list}")
        custom_check_point_output({'type':'update_info','node':'generate_response','data':contract_list})
        system_prompt = f"请根据以下合同信息回答用户提出的问题，请注意，严格参考合同信息，尽量不要遗漏提供的合同信息，因为这些合同已经由上游检测程序校准过，确认属于用户的询问范围，严禁虚构任何消息：\n{contract_list}\n"
        llm = ChatOpenAI(model="qwen-plus",api_key=qwen_api_key,base_url=qwen_base_url,temperature=0.01,max_completion_tokens=8000)
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

workflow.add_edge(START, "generate_search_words")
workflow.add_edge("generate_search_words", "vector_search")
workflow.add_edge("generate_search_words", "keyword_search")
workflow.add_edge("vector_search", "filter_contracts")
workflow.add_edge("keyword_search", "filter_contracts")
workflow.add_conditional_edges("filter_contracts", continue_check_contract_belong)
workflow.add_edge("check_contract_belong", "generate_response")
workflow.add_edge("generate_response", END)


# 编译工作流
graph = workflow.compile()

if __name__ == "__main__":
    # 测试用例
    query = "热电偶采购价格。"
    state = {"query": query,"messages":[]}
    result = graph.invoke(state)
    print(result)