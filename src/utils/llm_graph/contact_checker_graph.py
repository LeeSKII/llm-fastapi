from cmd import PROMPT
from typing import Literal,TypedDict, List, Optional,Dict
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.agents.structured_output import ToolStrategy
from textwrap import dedent
from datetime import datetime

load_dotenv()

# qwen3-max输出json解析嵌套对象有问题（输出的json文本不能正确由工具解析成对象），暂时推荐使用deepseek-chat或者本地部署的235b模型
api_key = os.getenv('QWEN_API_KEY')
base_url = os.getenv('QWEN_API_BASE_URL')
model_name = 'qwen3-max'

api_key = os.getenv('DEEPSEEK_API_KEY')
base_url = os.getenv('DEEPSEEK_API_BASE_URL')
model_name = 'deepseek-chat'

# local setting working as expect
# api_key = '123'
# base_url = 'http://192.168.0.166:8000/v1'
# model_name = 'Qwen3-235B'

class PartyInfo(TypedDict):
    """合同方详细信息"""
    party_name: str  # 方名称
    party_role: Literal['甲方', '乙方','丙方','其他']  # 方角色
    address: str  # 住所地
    legal_representative: str  # 法定代表人
    project_contact: str  # 项目联系人
    contact_phone: str  # 联系方式
    contact_address: str  # 通讯地址
    phone: Optional[str]  # 电话
    tax_number: Optional[str]  # 税号
    bank_name: Optional[str]  # 开户银行
    bank_account: Optional[str]  # 银行账户

class ContractBasicInfo(TypedDict):
    """合同基础信息"""
    contract_type: Literal['专业建设工程设计合同', '总承包合同', '设备销售合同', '咨询、设计及项目管理等服务合同'] # 业务分类
    business_type: Literal['境内工程项目合同', '境外工程项目合同', '非工程项目合同', '技术附件'] # 合同类型
    project_name: str  # 工程名称
    project_location: str  # 工程地点
    design_certificate_level: Optional[str]  # 设计证书等级
    employer: str  # 发包人/甲方
    designer: str  # 设计人/乙方
    party_info_list: List[PartyInfo]  # 合同方列表
    signing_date: datetime  # 签订日期
    contract_start_date: datetime  # 合同开始日期
    contract_end_date: datetime  # 合同结束日期
    contract_number: Optional[str]  # 合同编号

class ContractBasis(TypedDict):
    """合同签订依据"""
    laws: List[str]  # 法律法规依据
    regulations: List[str]  # 管理法规和规章
    approval_documents: List[str]  # 建设工程批准文件

class WorkBasis(TypedDict):
    """设计工作依据"""
    entrustment_document: Optional[str]  # 委托书或设计中标文件
    basic_data_provided: Optional[str]  # 甲方提交的基础资料
    technical_standards: List[str]  # 主要技术标准

class ProjectDetails(TypedDict):
    """项目详情"""
    project_scale: Optional[str]  # 工程规模
    project_stage: str  # 项目阶段
    project_investment: str  # 工程投资
    project_content: str  # 工程内容
    special_notes: Optional[str]  # 特别说明

class DataSubmissionSchedule(TypedDict):
    """资料提交安排"""
    submission_party: Literal['甲方', '乙方','丙方','其他']  # 提交方
    submission_content: str  # 提交内容
    submission_deadline: str  # 提交时间要求
    remarks: Optional[str]  # 备注

class DeliveryAchievement(TypedDict):
    """交付成果"""
    delivery_item: str  # 交付项名称
    delivery_stage: str   # 交付阶段
    delivery_content: str  # 交付内容描述
    copies: Optional[int]  # 交付份数
    delivery_method: Optional[str]  # 交付方式
    delivery_location: Optional[str]  # 交付地点
    prerequisite_conditions: str  # 交付前提条件
    delivery_period: str  # 交付时间要求
    acceptance_criteria: Optional[str]  # 验收标准
    quality_requirements: Optional[str]  # 质量要求
    special_notes: Optional[str]  # 特别说明

class FeeInfo(TypedDict):
    """费用信息"""
    total_fee: float  # 合同总费用（元）
    fee_in_words: str  # 大写金额
    tax_rate: str  # 增值税税率
    fee_basis: str  # 收费依据
    notes: Optional[str]  # 其它费用说明

class PaymentMilestone(TypedDict):
    """支付里程碑"""
    milestone_name: str  # 里程碑名称
    payment_condition: str  # 付款条件
    payment_amount: float  # 付款金额
    payment_in_words: str  # 大写金额
    payment_deadline: str  # 付款期限
    prerequisite: Optional[str]  # 前提条件（如发票要求）

class PartyResponsibility(TypedDict):
    """各方责任"""
    responsible_party: Literal['甲方', '乙方','丙方','其他']  # 责任方
    responsibility_item: str  # 责任事项
    responsibility_content: str  # 责任内容
    consequence: Optional[str]  # 不履行的后果

class IntellectualProperty(TypedDict):
    """知识产权条款"""
    ip_ownership_rule: str  # 所有权归属规则
    confidentiality_obligation: str  # 保密义务
    patent_application: str  # 专利申请规定
    existing_ip_usage: Optional[str]  # 现有知识产权使用

class ContractClause(TypedDict):
    """合同条款"""
    clause_title: str  # 条款标题/类型
    clause_content: str  # 补充协议规定

class DisputeResolution(TypedDict):
    """争议解决"""
    resolution_method: str  # 解决方式
    jurisdiction_court: Optional[str]  # 管辖法院
    applicable_law: str  # 适用法律

class ContractData(TypedDict):
    """完整的合同数据结构"""
    # 基础信息
    basic_info: ContractBasicInfo
    
    # 合同条款
    contract_basis: ContractBasis
    work_basis: WorkBasis
    project_details: ProjectDetails
    data_submission_schedule: List[DataSubmissionSchedule]
    delivery_achievements: List[DeliveryAchievement]
    fee_info: FeeInfo
    payment_milestones: List[PaymentMilestone]
    party_responsibilities: List[PartyResponsibility]
    intellectual_property: List[IntellectualProperty]
    dispute_resolution: DisputeResolution
    contract_clause: List[ContractClause]

class ContractAgentState(TypedDict):
    messages:list[Dict] | None
    contract_content:str
    basic_info: ContractBasicInfo # 合同基础信息
    contract_basis: ContractBasis # 合同签订依据
    work_basis: WorkBasis # 设计工作依据
    project_details: ProjectDetails # 项目详情
    data_submission_schedule: List[DataSubmissionSchedule] # 资料提交安排
    delivery_achievements: List[DeliveryAchievement] # 交付成果
    fee_info: FeeInfo # 费用信息
    payment_milestones: List[PaymentMilestone] # 支付里程碑
    party_responsibilities: List[PartyResponsibility] # 各方责任
    intellectual_property: List[IntellectualProperty] # 知识产权条款
    dispute_resolution: DisputeResolution # 争议解决
    contract_clause: List[ContractClause] # 合同条款
    

model = ChatOpenAI(api_key=api_key, base_url=base_url,temperature=0,model=model_name)

def general_prompt(contract_content:str)->str:
    prompt = dedent(f"""
    一步步分析合同并提取结构化数据，禁止遗漏，禁止虚构和推测未出现在合同文本中的内容和意图，按照提供的合同文本据实填写，按照用户提供的规则输出合法的json数据。

    合同文本: {contract_content}

    语言：简体中文。
    """)
    return prompt

def extract_contract_basic_data(state:ContractAgentState)->ContractAgentState:
    """提取合同基础信息"""
    contract_content = state['contract_content']
    prompt = general_prompt(contract_content)

    # 使用agent进行结构化输出，以前的json prompt方式在langchain中不建议使用，被移除了，因为未被证明比原生的api结构化输出或者是tool方式输出更可靠
    # model = ChatOpenAI(api_key=api_key, base_url=base_url,temperature=0,model=model_name)
    
    agent = create_agent(
        model=model,
        tools=[],
        response_format=ToolStrategy(ContractBasicInfo)
    )

    result = agent.invoke({'messages':prompt})
    structured_response = result['structured_response']
    return {'basic_info':structured_response}

def extract_contract_basis(state:ContractAgentState)->ContractAgentState:
    """提取合同签订依据"""
    contract_content = state['contract_content']
    prompt = general_prompt(contract_content)  
    agent = create_agent(model=model,response_format=ToolStrategy(ContractBasis))
    result = agent.invoke({'messages':prompt})
    structured_response = result['structured_response']
    return {'contract_basis':structured_response}

def extract_work_basis(state:ContractAgentState)->ContractAgentState:
    """提取设计工作依据"""
    contract_content = state['contract_content']
    prompt = general_prompt(contract_content)  
    agent = create_agent(model=model,response_format=ToolStrategy(WorkBasis))
    result = agent.invoke({'messages':prompt})
    structured_response = result['structured_response']
    return {'work_basis':structured_response}

def extract_project_details(state:ContractAgentState)->ContractAgentState:
    """提取项目详情"""
    contract_content = state['contract_content']
    prompt = general_prompt(contract_content)  
    agent = create_agent(model=model,response_format=ToolStrategy(ProjectDetails))
    result = agent.invoke({'messages':prompt})
    structured_response = result['structured_response']
    return {'project_details':structured_response}

def extract_data_submission_schedule(state:ContractAgentState)->ContractAgentState:
    """提取项目详情"""
    contract_content = state['contract_content']
    prompt = general_prompt(contract_content)  
    class DataSubmissionScheduleList(TypedDict):
        """资料提交安排"""
        data_submission_schedule: List[DataSubmissionSchedule]
    agent = create_agent(model=model,response_format=ToolStrategy(DataSubmissionScheduleList))
    result = agent.invoke({'messages':prompt})
    structured_response = result['structured_response']
    return {'data_submission_schedule':structured_response['data_submission_schedule']}

def extract_delivery_achievements(state:ContractAgentState)->ContractAgentState:
    """提取项目详情"""
    contract_content = state['contract_content']
    prompt = general_prompt(contract_content)  
    class DeliveryAchievementList(TypedDict):
        """交付成果列表"""
        delivery_achievements: List[DeliveryAchievement] # 交付成果
    agent = create_agent(model=model,response_format=ToolStrategy(DeliveryAchievementList))
    result = agent.invoke({'messages':prompt})
    structured_response = result['structured_response']
    return {'delivery_achievements':structured_response['delivery_achievements']}

def extract_delivery_achievements(state:ContractAgentState)->ContractAgentState:
    """提取项目详情"""
    contract_content = state['contract_content']
    prompt = general_prompt(contract_content)  
    class DeliveryAchievementList(TypedDict):
        """交付成果列表"""
        delivery_achievements: List[DeliveryAchievement] # 交付成果
    agent = create_agent(model=model,response_format=ToolStrategy(DeliveryAchievementList))
    result = agent.invoke({'messages':prompt})
    structured_response = result['structured_response']
    return {'delivery_achievements':structured_response['delivery_achievements']}

def extract_fee_info(state:ContractAgentState)->ContractAgentState:
    """提取合同费用信息"""
    contract_content = state['contract_content']
    prompt = general_prompt(contract_content)  
    agent = create_agent(model=model,response_format=ToolStrategy(FeeInfo))
    result = agent.invoke({'messages':prompt})
    structured_response = result['structured_response']
    return {'fee_info':structured_response}

def extract_payment_milestones(state:ContractAgentState)->ContractAgentState:
    """提取支付里程碑"""
    contract_content = state['contract_content']
    prompt = general_prompt(contract_content)  
    class PaymentMilestoneList(TypedDict):
        """支付里程碑列表"""
        payment_milestones: List[PaymentMilestone] # 支付里程碑
    agent = create_agent(model=model,response_format=ToolStrategy(PaymentMilestoneList))
    result = agent.invoke({'messages':prompt})
    structured_response = result['structured_response']
    return {'payment_milestones':structured_response['payment_milestones']}

def extract_party_responsibilities(state:ContractAgentState)->ContractAgentState:
    """提取各方责任"""
    contract_content = state['contract_content']
    prompt = general_prompt(contract_content)  
    class PartyResponsibilityList(TypedDict):
        """各方责任列表"""
        party_responsibilities: List[PartyResponsibility] # 各方责任
    agent = create_agent(model=model,response_format=ToolStrategy(PartyResponsibilityList))
    result = agent.invoke({'messages':prompt})
    structured_response = result['structured_response']
    return {'party_responsibilities':structured_response['party_responsibilities']}

def extract_intellectual_property(state:ContractAgentState)->ContractAgentState:
    """提取知识产权条款"""
    contract_content = state['contract_content']
    prompt = general_prompt(contract_content)  
    class IntellectualPropertyList(TypedDict):
        """知识产权条款列表"""
        intellectual_property: List[IntellectualProperty] # 知识产权条款
    agent = create_agent(model=model,response_format=ToolStrategy(IntellectualPropertyList))
    result = agent.invoke({'messages':prompt})
    structured_response = result['structured_response']
    return {'intellectual_property':structured_response['intellectual_property']}

def extract_dispute_resolution(state:ContractAgentState)->ContractAgentState:
    """提取争议解决"""
    contract_content = state['contract_content']
    prompt = general_prompt(contract_content)  
    agent = create_agent(model=model,response_format=ToolStrategy(DisputeResolution))
    result = agent.invoke({'messages':prompt})
    structured_response = result['structured_response']
    return {'dispute_resolution':structured_response}

def extract_contract_clause(state:ContractAgentState)->ContractAgentState:
    """提取合同条款"""
    contract_content = state['contract_content']
    prompt = general_prompt(contract_content)  
    class ContractClauseList(TypedDict):
        """合同条款列表"""
        contract_clause: List[ContractClause] # 合同条款
    agent = create_agent(model=model,response_format=ToolStrategy(ContractClauseList))
    result = agent.invoke({'messages':prompt})
    structured_response = result['structured_response']
    return {'contract_clause':structured_response['contract_clause']}

def gather_info(state:ContractAgentState)->ContractAgentState:
    # 汇聚节点
    return {}

workflow = StateGraph(ContractAgentState)
workflow.add_node('extract_contract_basic_data',extract_contract_basic_data)
workflow.add_node('extract_contract_basis',extract_contract_basis)
workflow.add_node('extract_work_basis',extract_work_basis)
workflow.add_node('extract_project_details',extract_project_details)
workflow.add_node('extract_data_submission_schedule',extract_data_submission_schedule)
workflow.add_node('extract_delivery_achievements',extract_delivery_achievements)
workflow.add_node('extract_fee_info',extract_fee_info)
workflow.add_node('extract_payment_milestones',extract_payment_milestones)
workflow.add_node('extract_party_responsibilities',extract_party_responsibilities)
workflow.add_node('extract_intellectual_property',extract_intellectual_property)
workflow.add_node('extract_dispute_resolution',extract_dispute_resolution)
workflow.add_node('extract_contract_clause',extract_contract_clause)
workflow.add_node('gather_info',gather_info)


workflow.add_edge(START,'extract_contract_basic_data')
workflow.add_edge(START,'extract_contract_basis')
workflow.add_edge(START,'extract_work_basis')
workflow.add_edge(START,'extract_project_details')
workflow.add_edge(START,'extract_data_submission_schedule')
workflow.add_edge(START,'extract_delivery_achievements')
workflow.add_edge(START,'extract_fee_info')
workflow.add_edge(START,'extract_payment_milestones')
workflow.add_edge(START,'extract_party_responsibilities')
workflow.add_edge(START,'extract_intellectual_property')
workflow.add_edge(START,'extract_dispute_resolution')
workflow.add_edge(START,'extract_contract_clause')

workflow.add_edge('extract_contract_basic_data','gather_info')
workflow.add_edge('extract_contract_basis','gather_info')
workflow.add_edge('extract_work_basis','gather_info')
workflow.add_edge('extract_project_details','gather_info')
workflow.add_edge('extract_data_submission_schedule','gather_info')
workflow.add_edge('extract_delivery_achievements','gather_info')
workflow.add_edge('extract_fee_info','gather_info')
workflow.add_edge('extract_payment_milestones','gather_info')
workflow.add_edge('extract_party_responsibilities','gather_info')
workflow.add_edge('extract_intellectual_property','gather_info')
workflow.add_edge('extract_dispute_resolution','gather_info')
workflow.add_edge('extract_contract_clause','gather_info')

workflow.add_edge('gather_info',END)

graph = workflow.compile()

if __name__ == '__main__':
    contract_content = "contract content"
    state:ContractAgentState = {'contract_content':contract_content}
    result = graph.invoke(state)
    print(result)