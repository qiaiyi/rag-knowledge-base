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
from vector_store import KnowledgeBase
from reranker import rerank
from config import CONFIG

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
async def rewrite_query(original_query, model=None):
    """
    异步利用 LLM 改写查询，提升检索效果
    """
    if model is None:
        model = CONFIG.LLM_MODEL

    prompt = f"""请将以下用户问题改写为更适合文档检索的形式。要求：
1. 如果问题已经清晰完整，直接返回原句。
2. 如果问题过于简短或口语化，请扩展为完整、清晰的查询，去除指代和模糊词。
3. 输出只包含改写后的查询，不要添加任何解释或额外文字。

原始问题：{original_query}
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
    
    async with httpx.AsyncClient(timeout=CONFIG.REQUEST_TIMEOUT) as client:
        resp = await client.post(f"{CONFIG.BASE_URL}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        rewritten = resp.json()["choices"][0]["message"]["content"].strip()
        return rewritten


# ============ 2. 非流式 LLM 生成答案（带重试） ============
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
    
    async with httpx.AsyncClient(timeout=CONFIG.LLM_TIMEOUT) as client:
        resp = await client.post(f"{CONFIG.BASE_URL}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ============ 3. 流式 LLM 生成答案（带重试 + 末尾附加来源标记） ============
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(is_retryable_exception)
)
async def stream_answer_with_context(question, context_chunks, model=None):
    """
    异步流式生成答案，复用 Prompt 模板
    流式传输结束后，附加特殊标记 __SOURCES__:JSON 用于前端解析来源
    """
    if model is None:
        model = CONFIG.LLM_MODEL

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

    async with httpx.AsyncClient(timeout=CONFIG.LLM_TIMEOUT) as client:
        async with client.stream("POST", f"{CONFIG.BASE_URL}/chat/completions", json=payload, headers=headers) as response:
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
        
        # ============ 流式结束后发送来源标记 ============
        sources_json = json.dumps(context_chunks, ensure_ascii=False)
        yield f"\n\n__SOURCES__:{sources_json}"


# ============ 4. 统一检索入口（纯逻辑，无重试，内部调用可重试函数） ============
async def retrieve_chunks(question, kb, top_k=CONFIG.TOP_K, score_threshold=CONFIG.SCORE_THRESHOLD,
                          use_rerank=CONFIG.USE_RERANK, rerank_top_n=CONFIG.RERANK_TOP_N,
                          use_query_rewrite=CONFIG.USE_QUERY_REWRITE):
    """
    异步统一检索入口：查询重写 → 向量检索（带阈值） → 重排序
    返回：已处理好的 context_chunks 列表（若无可检索内容则返回空列表）
    """
    # 1. 查询重写（可选）—— 内部自带重试
    if use_query_rewrite:
        try:
            rewritten_question = await rewrite_query(question)
        except RetryError as e:
            logger.warning(f"查询改写重试 3 次均失败，降级使用原问题: {e}")
            rewritten_question = question
    else:
        rewritten_question = question

    # 2. 向量检索（带分数过滤）
    results = await kb.search_with_scores(rewritten_question, top_k=top_k, score_threshold=score_threshold)
    if not results:
        return []

    chunks = [doc for doc, _ in results]

    # 3. 重排序（可选）
    if use_rerank and len(chunks) > 1:
        try:
            chunks = await rerank(rewritten_question, chunks, top_n=rerank_top_n)
        except Exception as e:
            logger.warning(f"重排序失败，降级使用向量检索结果: {e}")
            # 保持原结果

    return chunks


# ============ 5. 完整 RAG 流程（非流式） ============
async def answer_question(question, kb, top_k=CONFIG.TOP_K, score_threshold=CONFIG.SCORE_THRESHOLD,
                          use_rerank=CONFIG.USE_RERANK, rerank_top_n=CONFIG.RERANK_TOP_N,
                          use_query_rewrite=CONFIG.USE_QUERY_REWRITE):
    """
    异步完整 RAG 流程：检索 + 生成（非流式）
    """
    chunks = await retrieve_chunks(question, kb, top_k, score_threshold, use_rerank, rerank_top_n, use_query_rewrite)
    if not chunks:
        return "知识库中没有找到相关内容。", []
    
    try:
        answer = await ask_with_context(question, chunks)
    except RetryError as e:
        logger.error(f"LLM 生成重试 3 次均失败: {e}")
        answer = "生成答案时遇到错误，请稍后重试。"
    
    return answer, chunks