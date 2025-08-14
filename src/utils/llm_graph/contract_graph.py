from langgraph.graph import StateGraph, START, END
from typing import TypedDict,Annotated,List,Optional,Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage
import os
from pydantic import BaseModel,Field
from langchain.output_parsers import PydanticOutputParser
from openai import OpenAI
import lancedb
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

qwen_api_key = os.getenv("QWEN_API_KEY")
qwen_base_url = os.getenv("QWEN_API_BASE_URL")

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
    messages:List[dict]
    response:str
    error:str

def generate_search_words(state: State) -> State:
    try:
      llm = ChatOpenAI(model="qwen-plus",api_key=qwen_api_key,base_url=qwen_base_url,temperature=0.01)
      llm_with_structured_output = llm.with_structured_output(SearchKeyWords).with_retry(stop_after_attempt=3)
      parser = PydanticOutputParser(pydantic_object=SearchKeyWords)
      format_instructions = parser.get_format_instructions()
      system_prompt = "你是一个合同查询的助手，用户提出的查询需求可能是关于某些项目的也可能是关于某些设备的，请将用户的查询分解成不同的查询关键字，你只能提取用户提供的查询关键字，这意味着你绝对禁止虚拟项目或者设备关键字，只能从用户的原始查询中提取，注意，你需要提取的是关键字，这个关键字用于执行SQL的精确搜索，所以一定要保持简洁，否则会导致严重的业务问题。需要按照提供的json格式进行输出：\n{format_instructions}\n"
      response = llm_with_structured_output.invoke([SystemMessage(system_prompt.format(format_instructions=format_instructions)),*state["messages"], HumanMessage(state["query"])])
      return {"meta_search_query_keyword": response}
    except Exception as e:
        return {"meta_search_query_keyword": SearchKeyWords(project_key_words=None,equipments_key_words=None),"error":"generate_search_words.\n"+str(e)}
    
def vector_search(state: State)->State:
    try:
       if state["meta_search_query_keyword"].project_key_words is not None:
           df_list = []
           db = lancedb.connect(r"C:\Lee\work\contract\csv\v3\contract_full_lancedb") 
           table = db.open_table("contract_table")
           for word in state["meta_search_query_keyword"].project_key_words:
               embedding = get_embedding(word)
               search_vector_results = table.search(embedding,vector_column_name="meta_vector").limit(5).to_pandas()
               df_list.append(search_vector_results)
           search_results = pd.concat(df_list, ignore_index=True)
           # 根据组合列去重（保留第一个出现的值）
           final_df = search_results.drop_duplicates(subset=['contact_no', 'project_name'])
           return {"vector_search_contract": final_df}
    except Exception as e:
        return {"vector_search_contract": None,"error":"vector_search.\n"+str(e)}
    
def keyword_search(state: State)->State:
    try:
        df_list = []
        db = lancedb.connect(r"C:\Lee\work\contract\csv\v3\contract_full_lancedb") 
        table = db.open_table("contract_table")
        merged_list = (state["meta_search_query_keyword"].project_key_words or []) + (state["meta_search_query_keyword"].equipments_key_words or [])
        
        for word in merged_list:
            search_like_results = table.search().where(f"doc LIKE '%{word}%'").limit(5).to_pandas()
            df_list.append(search_like_results)

        search_results = pd.concat(df_list, ignore_index=True)
        # 根据组合列去重（保留第一个出现的值）
        final_df = search_results.drop_duplicates(subset=['contact_no', 'project_name'])
           
        return {"keyword_search_contract": final_df}
    except Exception as e:
        return {"keyword_search_contract": None,"error":"keyword_search.\n"+str(e)}

def filter_contracts(state: State)->State:
    '''进行合同的过滤处理'''
    try:
        # 合并两个来源的df，排除重复数据
        final_df = pd.concat([state["vector_search_contract"],state["keyword_search_contract"]], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['contact_no', 'project_name'])
        # LLM过滤掉不属于有效查询范围的合同 TODO
        return {"filtered_contracts": final_df.to_dict(orient="records")}
    except Exception as e:
        return {"filtered_contracts": [],"error":"filter_contracts.\n"+str(e)}

def generate_response(state: State)->State:
    try:
        if state["filtered_contracts"] is not None and len(state["filtered_contracts"]) > 0:
            contract_list = []
            for contract in state["filtered_contracts"]:
                contract_list.append(f"合同元数据：{contract['contract_meta']}\n合同设计设备:{contract['equipment_table']}\n============\n")
        system_prompt = f"请根据以下合同信息回答用户提出的问题，请注意，严格参考合同信息，严禁虚构任何消息：\n{contract_list}\n"
        llm = ChatOpenAI(model="qwen-plus",api_key=qwen_api_key,base_url=qwen_base_url,temperature=0.01)
        response = llm.invoke([SystemMessage(system_prompt)],*state["messages"], HumanMessage(state["query"]))
        return {"response": response.content}
    except Exception as e:
        return {"response": None,"error":"generate_response.\n"+str(e)}

# 构建 LangGraph 工作流
workflow = StateGraph(State)
workflow.add_node("generate_search_words", generate_search_words)
workflow.add_node("vector_search", vector_search)
workflow.add_node("keyword_search", keyword_search)
workflow.add_node("filter_contracts", filter_contracts)
workflow.add_node("generate_response", generate_response)

workflow.add_edge(START, "generate_search_words")
workflow.add_edge("generate_search_words", "vector_search")
workflow.add_edge("generate_search_words", "keyword_search")
workflow.add_edge("vector_search", "filter_contracts")
workflow.add_edge("keyword_search", "filter_contracts")
workflow.add_edge("filter_contracts", "generate_response")
workflow.add_edge("generate_response", END)


# 编译工作流
graph = workflow.compile()

if __name__ == "__main__":
    # 测试用例
    query = "热电偶采购价格"
    state = {"query": query,"messages":[]}
    result = graph.invoke(state)
    print(result)