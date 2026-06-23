import httpx
import os
import logging
from dotenv import load_dotenv
from config import CONFIG

load_dotenv()
logger = logging.getLogger(__name__)


async def rerank(query, documents, top_n=3):
    """
    异步使用硅基流动的 BGE-reranker 对文档片段进行重排序
    :param query: 用户问题
    :param documents: 候选文档列表（一般来自向量检索的 top-k）
    :param top_n: 返回的最相关文档数量
    :return: 重排序后的文档列表（按相关性降序）
    """
    if not documents:
        return []

    url = f"{CONFIG.SF_BASE_URL}/rerank"
    headers = {
        "Authorization": f"Bearer {CONFIG.SF_API_KEY}",  # 修正：原为 SF_BASE_API，应为 SF_API_KEY
        "Content-Type": "application/json"
    }
    payload = {
        "model": CONFIG.RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n
    }
    try:
        async with httpx.AsyncClient(timeout=CONFIG.REQUEST_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            sorted_results = sorted(data["results"], key=lambda x: x["relevance_score"], reverse=True)
            ranked_docs = [documents[item["index"]] for item in sorted_results]
            return ranked_docs[:top_n]
    except httpx.TimeoutException as e:
        logger.warning(f"Rerank 请求超时，降级至原始向量检索结果: {e}")
        return documents[:top_n]
    except httpx.HTTPStatusError as e:
        logger.warning(f"Rerank HTTP 错误 {e.response.status_code}，降级至原始向量检索结果: {e}")
        return documents[:top_n]
    except Exception as e:
        logger.warning(f"Rerank 调用失败，降级至原始向量检索结果: {e}")
        return documents[:top_n]