import ast
import operator

from langchain_core.tools import tool
from tavily import AsyncTavilyClient

from backend.config import CONFIG
from backend.core.vector_store import KnowledgeBase
from backend.core.rag_chain import retrieve_chunks

# ========== 初始化外部客户端（工具依赖） ==========
_tavily_client = AsyncTavilyClient(api_key=CONFIG.TAVILY_API_KEY)


# ========== 工具 1：知识库检索（kb 通过闭包注入，不暴露给模型） ==========
def make_retrieve_kb(kb: KnowledgeBase):
    """为指定知识库生成检索工具。模型只看到 query 参数，kb 自动绑定。"""

    @tool
    async def retrieve_kb(query: str) -> str:
        """从公司内部知识库检索文档片段（适用于制度、产品手册、内部FAQ）。"""
        try:
            chunks = await retrieve_chunks(
                query, kb,
                top_k=CONFIG.TOP_K,
                score_threshold=CONFIG.SCORE_THRESHOLD,
                use_rerank=CONFIG.USE_RERANK,
                rerank_top_n=CONFIG.RERANK_TOP_N,
                use_query_rewrite=CONFIG.USE_QUERY_REWRITE,
            )
            if not chunks:
                return "知识库中未找到相关信息。"
            return "\n\n".join(chunks)
        except Exception as e:
            return f"知识库检索失败: {e}"

    return retrieve_kb


# ========== 工具 2：网络搜索 ==========
@tool
async def web_search(query: str, topic: str = None, time_range: str = None) -> str:
    """搜索实时外部信息（新闻、天气、股票、近期事件）。"""
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


# ========== 工具 3：计算器（保持纯函数，另包一层给 agent 用，便于测试） ==========
def calculator(expression: str) -> str:
    """
    安全计算数学表达式。

    白名单只允许数字与加减乘除取模运算符，并用 ast 解析成语法树递归求值，
    从根上杜绝 eval() 带来的代码注入风险。
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
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_fn = _BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        return op_fn(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


@tool
def calculator_tool(expression: str) -> str:
    """计算数学表达式，如 '123*456'、'(1+2)*3'。"""
    return calculator(expression)