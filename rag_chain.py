import httpx
import os
import json
import logging
from dotenv import load_dotenv
from vector_store import KnowledgeBase
from document_loader import load_document
from text_splitter import split_text
from reranker import rerank
from config import CONFIG

load_dotenv()
logger = logging.getLogger(__name__)

# Prompt 模板已移至 CONFIG.RAG_PROMPT_TEMPLATE，此处不再重复定义


async def rewrite_query(original_query, model=None):
    """
    异步利用 LLM 改写查询，提升检索效果
    :param original_query: 原始用户问题
    :param model: 使用的 LLM 模型名称
    :return: 改写后的查询字符串
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
    try:
        async with httpx.AsyncClient(timeout=CONFIG.REQUEST_TIMEOUT) as client:
            resp = await client.post(f"{CONFIG.BASE_URL}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            rewritten = resp.json()["choices"][0]["message"]["content"].strip()
            return rewritten
    except Exception as e:
        logger.warning(f"查询重写失败: {e}")
        return original_query  # 降级：返回原问题


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
    try:
        async with httpx.AsyncClient(timeout=CONFIG.LLM_TIMEOUT) as client:
            resp = await client.post(f"{CONFIG.BASE_URL}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        return "生成答案时遇到错误。"


async def retrieve_chunks(question, kb, top_k=CONFIG.TOP_K, score_threshold=CONFIG.SCORE_THRESHOLD,
                          use_rerank=CONFIG.USE_RERANK, rerank_top_n=CONFIG.RERANK_TOP_N,
                          use_query_rewrite=CONFIG.USE_QUERY_REWRITE):
    """
    异步统一检索入口：查询重写 → 向量检索（带阈值） → 重排序
    返回：已处理好的 context_chunks 列表（若无可检索内容则返回空列表）
    """
    # 1. 查询重写（可选）
    if use_query_rewrite:
        rewritten_question = await rewrite_query(question)
    else:
        rewritten_question = question

    # 2. 向量检索（带分数过滤）— search_with_scores 现在是异步的
    results = await kb.search_with_scores(rewritten_question, top_k=top_k, score_threshold=score_threshold)
    if not results:
        return []

    chunks = [doc for doc, _ in results]

    # 3. 重排序（可选）— rerank 现在是异步的
    if use_rerank and len(chunks) > 1:
        chunks = await rerank(rewritten_question, chunks, top_n=rerank_top_n)

    return chunks


async def answer_question(question, kb, top_k=CONFIG.TOP_K, score_threshold=CONFIG.SCORE_THRESHOLD,
                          use_rerank=CONFIG.USE_RERANK, rerank_top_n=CONFIG.RERANK_TOP_N,
                          use_query_rewrite=CONFIG.USE_QUERY_REWRITE):
    """
    异步完整 RAG 流程：检索 + 生成
    """
    chunks = await retrieve_chunks(question, kb, top_k, score_threshold, use_rerank, rerank_top_n, use_query_rewrite)
    if not chunks:
        return "知识库中没有找到相关内容。", []
    answer = await ask_with_context(question, chunks)
    return answer, chunks


async def stream_answer_with_context(question, context_chunks, model=None):
    """
    异步流式生成答案，复用 Prompt 模板
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

    try:
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
    except Exception as e:
        logger.error(f"流式 LLM 调用失败: {e}")
        yield "生成答案时遇到错误。"