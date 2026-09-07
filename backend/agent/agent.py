# agent.py
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from backend.config import CONFIG
from backend.core.vector_store import KnowledgeBase
from backend.agent.tools import make_retrieve_kb, web_search, calculator_tool


def build_react_agent(kb: KnowledgeBase):
    """
    ReAct Agent：用 LangChain(LangModel) + LangGraph(create_agent) 搭建。

    - LLM：ChatOpenAI（复用 RAG 那份硅基流动/OpenAI 兼容配置）
    - 工具：知识库检索(注入 kb) + 网络搜索 + 计算器，均由 LangChain @tool 声明
    - 编排：create_agent 自动处理“思考→调用工具→看结果→再思考”的循环与终止

    @tool / create_agent 会自动生成工具 schema、派发工具调用、解析 tool_calls，
    因此无需再手写 TOOL_SCHEMAS / TOOL_MAP / tool_node / StateGraph。
    """
    llm = ChatOpenAI(
        model=CONFIG.LLM_MODEL,
        temperature=CONFIG.TEMPERATURE,
        api_key=CONFIG.API_KEY,
        base_url=CONFIG.BASE_URL,
        max_tokens=CONFIG.MAX_TOKENS,
        timeout=CONFIG.LLM_TIMEOUT,
    )
    tools = [make_retrieve_kb(kb), web_search, calculator_tool]

    return create_agent(
        llm,
        tools,
        system_prompt=CONFIG.SYSTEM_PROMPT,
    )