import json
import ast
import operator
import asyncio
from typing import Dict, Callable, Any, Awaitable
from openai import AsyncOpenAI
from tavily import AsyncTavilyClient
from config import CONFIG
from vector_store import KnowledgeBase
from rag_chain import retrieve_chunks

# ========== 初始化外部客户端（工具依赖） ==========
# 注意：这里初始化是为了让工具函数能直接使用，但建议在函数内部按需调用
_tavily_client = AsyncTavilyClient(api_key=CONFIG.TAVILY_API_KEY)


# ========== 1. 工具函数实现 ==========
async def retrieve_kb(query: str, kb: KnowledgeBase) -> str:
    """从内部知识库检索相关文档片段（异步）"""
    try:
        chunks = await retrieve_chunks(
            query, kb,
            top_k=CONFIG.TOP_K,
            score_threshold=CONFIG.SCORE_THRESHOLD,
            use_rerank=CONFIG.USE_RERANK,
            rerank_top_n=CONFIG.RERANK_TOP_N,
            use_query_rewrite=CONFIG.USE_QUERY_REWRITE
        )
        if not chunks:
            return "知识库中未找到相关信息。"
        return "\n\n".join(chunks)
    except Exception as e:
        return f"知识库检索失败: {e}"


async def web_search(
    query: str,
    topic: str = None,
    time_range: str = None,
) -> str:
    """使用 Tavily 搜索引擎获取实时信息（异步）"""
    try:
        response = await _tavily_client.search(
            query=query,
            search_depth=CONFIG.TAVILY_SEARCH_DEPTH,
            max_results=CONFIG.TAVILY_MAX_RESULTS,
            topic=topic or CONFIG.TAVILY_TOPIC or None,
            time_range=time_range or CONFIG.TAVILY_TIME_RANGE or None,
        )
        results = []
        for result in response.get("results", []):
            published = result.get("published_date", "")
            date_line = f"\n日期: {published}" if published else ""
            results.append(
                f"标题: {result.get('title', 'N/A')}\n"
                f"内容: {result.get('content', '')}\n"
                f"来源: {result.get('url', '')}"
                f"{date_line}"
            )
        if not results:
            return "未搜索到相关信息。"
        return "\n\n".join(results)
    except Exception as e:
        return f"搜索失败: {e}"


def calculator(expression: str) -> str:
    """
    执行数学运算（同步，CPU密集型）

    安全实现：用 ast 模块把表达式解析成语法树，只允许数字和
    加减乘除取模运算，杜绝 eval() 带来的代码注入风险。
    """
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含非法字符。"
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


# ---- AST 安全求值器：只处理数字和算术运算，遇到其他节点直接报错 ----
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}

def _safe_eval(node):
    """递归遍历 AST 节点，只允许常量和二元/一元算术运算"""
    if isinstance(node, ast.Constant):          # 数字字面量
        return node.value
    if isinstance(node, ast.BinOp):             # a + b、a * b 等
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_fn = _BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        return op_fn(left, right)
    if isinstance(node, ast.UnaryOp):           # 负号 -a
        operand = _safe_eval(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


# ========== 2. 工具元数据（OpenAI Function Schema） ==========
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_kb",
            "description": "从公司内部知识库检索文档（适用于制度、产品手册、内部FAQ）",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用户问题，如：'公司放假安排'"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
        "description": "搜索实时外部信息（新闻、天气、股票、近期事件）",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "topic": {
                    "type": "string",
                    "enum": ["general", "news", "finance"],
                    "description": "搜索主题：general=通用, news=新闻, finance=金融。搜索新闻类内容请用 news"
                },
                "time_range": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year"],
                    "description": "时间范围筛选：day=一天内, week=一周内, month=一月内, year=一年内"
                }
            },
            "required": ["query"]
        }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 '123*456'"}
                },
                "required": ["expression"]
            }
        }
    }
]

# ========== 3. 工具注册表（函数名 -> 可调用对象） ==========
# 注意：异步函数直接放进去，同步函数也放进去，由调用方区分执行
TOOL_MAP: Dict[str, Callable[..., Any]] = {
    "retrieve_kb": retrieve_kb,
    "web_search": web_search,
    "calculator": calculator,
}

# ========== 4. 便捷工具函数（供 agent 调用） ==========
def get_tool_schemas():
    """返回工具 Schema 列表"""
    return TOOL_SCHEMAS

def get_tool_map():
    """返回工具注册表"""
    return TOOL_MAP

# 标记哪些工具是异步的（便于 agent 判断是否要 await）
ASYNC_TOOL_NAMES = {"retrieve_kb", "web_search"}
