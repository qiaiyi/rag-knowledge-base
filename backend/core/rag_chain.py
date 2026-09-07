import json
import httpx
import logging
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception,
    RetryError
)
from dotenv import load_dotenv
from backend.core.vector_store import KnowledgeBase
from backend.core.reranker import rerank
from backend.core.http_client import get_client
from backend.config import CONFIG

load_dotenv()
logger = logging.getLogger(__name__)


# ============ 定义“哪些异常需要重试”的判断函数 ============
def is_retryable_exception(exception):
    """
    判断异常是否值得重试：
    - 超时（TimeoutException）-> 重试
    - HTTP 状态码 429（限流）、502/503/504（服务端临时故障）-> 重试
    - 其他 4xx 错误（如 400 参数错误）-> 不重试，直接报错
    """
    if isinstance(exception, httpx.TimeoutException):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        # 只对服务端临时错误和限流重试
        return exception.response.status_code in [429, 502, 503, 504]
    return False


# ============ 1. 查询改写（带重试） ============
@retry(
    stop=stop_after_attempt(3),                      # 最多重试 3 次
    wait=wait_exponential(multiplier=1, min=1, max=10),  # 等 1s、2s、4s（最长 10s）
    retry=retry_if_exception(is_retryable_exception)     # 只对特定异常重试
)
async def rewrite_query(original_query, model=None, history=None):
    """
    异步利用 LLM 改写查询，提升检索效果。
    传入 history（[{"role","content"}, ...]）时，会结合最近对话消解指代，
    把"它的作用是什么"这类追问改写为独立完整的检索查询。
    """
    if model is None:
        model = CONFIG.LLM_MODEL

    history_block = ""
    if history:
        recent = history[-CONFIG.HISTORY_MAX_MESSAGES:]
        lines = [f"{m['role']}: {m['content']}" for m in recent]
        history_block = "【最近对话】\n" + "\n".join(lines) + "\n\n"

    prompt = f"""请将以下用户问题改写为更适合文档检索的形式。要求：
1. 结合最近对话历史（如有），将问题中的指代（如"它"、"这个"）替换为具体对象，扩展为独立、完整、清晰的查询。
2. 如果问题已经清晰完整，直接返回原句。
3. 输出只包含改写后的查询，不要添加任何解释或额外文字。

{history_block}原始问题：{original_query}
改写："""
    
    headers = {
        "Authorization": f"Bearer {CONFIG.API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": CONFIG.TEMPERATURE
    }
    
    client = get_client()
    resp = await client.post(f"{CONFIG.BASE_URL}/chat/completions", json=payload, headers=headers, timeout=CONFIG.REQUEST_TIMEOUT)
    resp.raise_for_status()
    rewritten = resp.json()["choices"][0]["message"]["content"].strip()
    return rewritten


# ============ 2. 上下文预算截断 ============
def _estimate_tokens(text):
    """保守估算文本的 token 数：中文场景 1 字符 ≈ 1 token。
    若需精确 token 可以换成 tiktoken.encode"""
    return len(text)


def truncate_context(chunks, max_tokens):
    """
    按 token 预算从头部起截断检索上下文（保序，至少保留 1 条）。
    因只丢弃尾部、不重排，保留片段仍与 hits 前 N 条一一对应，引用编号不失效。
    :param chunks: 检索命中的文本内容列表
    :param max_tokens: 允许的最大 token 数
    :return: 截断后的文本列表
    """
    if max_tokens <= 0:
        return chunks[:1] if chunks else []
    budget = 0
    kept = []
    for chunk in chunks:
        if kept and budget + _estimate_tokens(chunk) > max_tokens:
            break  # 超预算即停，保证至少保留 1 条
        budget += _estimate_tokens(chunk)
        kept.append(chunk)
    return kept


# ============ 3. 非流式 LLM 生成答案（带重试） ============
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(is_retryable_exception)
)
async def ask_with_context(question, context_chunks, model=None):
    """
    异步根据检索到的上下文生成答案（带引用编号）
    """
    if model is None:
        model = CONFIG.LLM_MODEL
    context_chunks = truncate_context(context_chunks, CONFIG.MAX_CONTEXT_TOKENS)

    numbered_chunks = [f"[{i}] {chunk}" for i, chunk in enumerate(context_chunks, 1)]
    context_text = "\n\n".join(numbered_chunks)
    prompt = CONFIG.RAG_PROMPT_TEMPLATE.format(context=context_text, question=question)

    headers = {
        "Authorization": f"Bearer {CONFIG.API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": CONFIG.TEMPERATURE
    }
    
    client = get_client()
    resp = await client.post(f"{CONFIG.BASE_URL}/chat/completions", json=payload, headers=headers, timeout=CONFIG.LLM_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ============ 3. 流式 LLM 生成答案（带重试，产出正文增量） ============
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(is_retryable_exception)
)
async def stream_answer_with_context(question, context_chunks, model=None):
    """
    异步流式生成答案，复用 Prompt 模板。
    只产出正文文本增量；来源元数据由调用方在上游检索完成后单独下发（真 SSE 事件）。
    """
    if model is None:
        model = CONFIG.LLM_MODEL
    context_chunks = truncate_context(context_chunks, CONFIG.MAX_CONTEXT_TOKENS)

    numbered_chunks = [f"[{i}] {chunk}" for i, chunk in enumerate(context_chunks, 1)]
    context_text = "\n\n".join(numbered_chunks)
    prompt = CONFIG.RAG_PROMPT_TEMPLATE.format(context=context_text, question=question)

    headers = {"Authorization": f"Bearer {CONFIG.API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": CONFIG.TEMPERATURE,
        "stream": True
    }

    client = get_client()
    async with client.stream("POST", f"{CONFIG.BASE_URL}/chat/completions", json=payload, headers=headers, timeout=CONFIG.LLM_TIMEOUT) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line:
                continue
            if line.startswith('data: '):
                data_str = line[6:]
                if data_str == '[DONE]':
                    break
                try:
                    chunk_data = json.loads(data_str)
                    delta = chunk_data['choices'][0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue


# ============ 4. 统一检索入口（纯逻辑，无重试，内部调用可重试函数） ============
async def retrieve_chunks_detailed(question, kb, top_k=CONFIG.TOP_K, score_threshold=CONFIG.SCORE_THRESHOLD,
                                   use_rerank=CONFIG.USE_RERANK, rerank_top_n=CONFIG.RERANK_TOP_N,
                                   use_query_rewrite=CONFIG.USE_QUERY_REWRITE, history=None):
    """
    异步统一检索入口：查询重写（可结合对话历史） → 向量检索（带阈值） → 重排序
    返回：[{"content", "score", "source"}, ...]（若无可检索内容则返回空列表）
    """
    # 1. 查询重写（可选）—— 内部自带重试
    if use_query_rewrite:
        try:
            rewritten_question = await rewrite_query(question, history=history)
        except RetryError as e:
            logger.warning(f"查询改写重试 3 次均失败，降级使用原问题: {e}")
            rewritten_question = question
    else:
        rewritten_question = question

    # 2. 向量检索（带分数过滤）
    hits = await kb.search_with_scores(rewritten_question, top_k=top_k, score_threshold=score_threshold)
    if not hits:
        return []

    # 3. 重排序（可选）
    if use_rerank and len(hits) > 1:
        try:
            chunks = [h["content"] for h in hits]
            reranked = await rerank(rewritten_question, chunks, top_n=rerank_top_n)
            # 按 rerank 结果重新映射回带元数据的命中项
            by_content = {h["content"]: h for h in hits}
            hits = [by_content[c] for c in reranked if c in by_content]
        except Exception as e:
            logger.warning(f"重排序失败，降级使用向量检索结果: {e}")
            # 保持原结果

    return hits


async def retrieve_chunks(question, kb, top_k=CONFIG.TOP_K, score_threshold=CONFIG.SCORE_THRESHOLD,
                          use_rerank=CONFIG.USE_RERANK, rerank_top_n=CONFIG.RERANK_TOP_N,
                          use_query_rewrite=CONFIG.USE_QUERY_REWRITE, history=None):
    """
    兼容接口：只返回检索到的文本内容列表（tools.py 等使用）
    """
    hits = await retrieve_chunks_detailed(
        question, kb, top_k, score_threshold, use_rerank, rerank_top_n, use_query_rewrite, history
    )
    return [h["content"] for h in hits]


# ============ 5. 完整 RAG 流程（非流式） ============
async def answer_question(question, kb, top_k=CONFIG.TOP_K, score_threshold=CONFIG.SCORE_THRESHOLD,
                          use_rerank=CONFIG.USE_RERANK, rerank_top_n=CONFIG.RERANK_TOP_N,
                          use_query_rewrite=CONFIG.USE_QUERY_REWRITE, history=None):
    """
    异步完整 RAG 流程：检索（可结合多轮历史） + 生成（非流式）
    返回：(answer, sources)，sources 为 [{"content", "score", "source"}, ...]
    """
    hits = await retrieve_chunks_detailed(
        question, kb, top_k, score_threshold, use_rerank, rerank_top_n, use_query_rewrite, history
    )
    if not hits:
        return "知识库中没有找到相关内容。", []

    try:
        answer = await ask_with_context(question, [h["content"] for h in hits])
    except RetryError as e:
        logger.error(f"LLM 生成重试 3 次均失败: {e}")
        answer = "生成答案时遇到错误，请稍后重试。"

    return answer, hits