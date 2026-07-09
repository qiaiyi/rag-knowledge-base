# agent.py
import json
import operator
import asyncio
from typing import Annotated, Literal
from typing_extensions import TypedDict
from openai import AsyncOpenAI
from langgraph.graph import StateGraph, START, END
from config import CONFIG
from vector_store import KnowledgeBase
from tools import get_tool_schemas, get_tool_map, ASYNC_TOOL_NAMES


# ========== 1. 初始化 LLM 客户端 ==========
llm_client = AsyncOpenAI(
    api_key=CONFIG.API_KEY,
    base_url=CONFIG.BASE_URL,
    timeout=CONFIG.LLM_TIMEOUT
)

TOOL_SCHEMAS = get_tool_schemas()
TOOL_MAP = get_tool_map()


# ========== 2. 定义状态（使用 operator.add 自动追加消息列表） ==========
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


# ========== 3. 将消息统一转为 OpenAI API 兼容的 dict 格式 ==========
def msg_to_openai_dict(msg):
    """将 LangChain BaseMessage 对象或 plain dict 转为 OpenAI API 兼容的 dict。"""
    if not hasattr(msg, "type"):
        # 已经是 dict
        return msg

    # LangChain 消息对象
    role_map = {"human": "user", "ai": "assistant", "tool": "tool"}
    d = {"role": role_map.get(msg.type, msg.type), "content": msg.content}

    if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["args"], ensure_ascii=False) if isinstance(tc["args"], dict) else str(tc["args"]),
                },
            }
            for tc in msg.tool_calls
        ]

    if msg.type == "tool" and hasattr(msg, "tool_call_id"):
        d["tool_call_id"] = msg.tool_call_id

    return d


# ========== 4. 节点函数 ==========
async def agent_node(state: AgentState):
    messages = state["messages"]

    system_msg = {"role": "system", "content": CONFIG.SYSTEM_PROMPT}

    allowed_fields = {"role", "content", "tool_calls", "tool_call_id"}
    cleaned = []
    for msg in messages:
        d = msg_to_openai_dict(msg)
        clean = {k: v for k, v in d.items() if k in allowed_fields}
        if "content" not in clean:
            clean["content"] = None
        cleaned.append(clean)

    full_messages = [system_msg] + cleaned

    try:
        response = await llm_client.chat.completions.create(
            model=CONFIG.LLM_MODEL,
            messages=full_messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=CONFIG.TEMPERATURE,
            max_tokens=CONFIG.MAX_TOKENS,
        )
    except Exception as e:
        print(f"LLM API 调用失败: {e}")
        raise

    msg = response.choices[0].message
    new_msg = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        new_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return {"messages": [new_msg]}


async def tool_node(state: AgentState, kb: KnowledgeBase):
    messages = state["messages"]
    last_msg = messages[-1]
    last_msg = msg_to_openai_dict(last_msg)

    if not last_msg.get("tool_calls"):
        return {}

    tool_results = []
    for tc in last_msg["tool_calls"]:
        func_name = tc["function"]["name"]

        try:
            args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": f"参数解析错误: {tc['function']['arguments']}",
            })
            continue

        func = TOOL_MAP.get(func_name)
        if func is None:
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": f"未知工具: {func_name}",
            })
            continue

        try:
            if func_name in ASYNC_TOOL_NAMES:
                if func_name == "retrieve_kb":
                    result = await func(args.get("query", ""), kb)
                else:
                    result = await func(args.get("query", ""))
            else:
                result = await asyncio.to_thread(func, **args)
        except Exception as e:
            result = f"工具执行错误: {e}"

        tool_results.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": str(result),
        })

    return {"messages": tool_results}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_msg = msg_to_openai_dict(messages[-1])
    if last_msg.get("tool_calls"):
        return "tools"
    return "__end__"


# ========== 5. 构建图 ==========
def build_react_agent(kb: KnowledgeBase):
    async def tool_node_wrapper(state: AgentState):
        return await tool_node(state, kb)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_wrapper)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "__end__": END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()
