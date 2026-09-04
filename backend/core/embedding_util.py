import httpx
import os
from dotenv import load_dotenv
import logging
from backend.config import CONFIG
from backend.core.http_client import get_client

load_dotenv()
logger = logging.getLogger(__name__)

# 单次批量请求的最大文本数（硅基流动 /embeddings 对 batch 大小有上限，保守取 64）
EMBEDDING_BATCH_SIZE = 64


class EmbeddingError(Exception):
    """当 Embedding API 调用失败时抛出的自定义异常"""
    pass


async def get_embeddings(texts):
    """
    异步批量调用硅基流动 embedding API，按输入顺序返回向量列表
    :param texts: 文本列表（非空）
    :return: 向量列表，与 texts 一一对应
    :raises EmbeddingError: API 调用失败、响应解析失败或数量不匹配时抛出
    :raises ValueError: 输入为空时抛出
    """
    if not texts:
        raise ValueError("输入文本列表为空，无法生成向量")
    if any(not t or not t.strip() for t in texts):
        raise ValueError("输入文本为空，无法生成向量")

    url = f"{CONFIG.SF_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {CONFIG.SF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": CONFIG.EMBEDDING_MODEL,
        "input": texts
    }

    try:
        client = get_client()
        resp = await client.post(url, json=payload, headers=headers, timeout=CONFIG.REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        # 校验返回结构是否完整（防御性编程）
        if not data.get("data"):
            raise EmbeddingError("API 返回数据结构异常，缺少 embedding 字段")

        embeddings = [None] * len(texts)
        for item in data["data"]:
            embedding = item.get("embedding")
            if not embedding:
                raise EmbeddingError("API 返回数据结构异常，缺少 embedding 字段")
            embeddings[item["index"]] = embedding

        actual = len([e for e in embeddings if e is not None])
        if actual != len(texts):
            raise EmbeddingError(f"API 返回向量数量不匹配: 期望 {len(texts)} 条，实际 {actual} 条")

        return embeddings

    except httpx.TimeoutException as e:
        logger.error(f"Embedding API 请求超时: {e}")
        raise EmbeddingError(f"Embedding 服务超时: {e}") from e
    except httpx.HTTPStatusError as e:
        logger.error(f"Embedding API HTTP 错误: {e}")
        raise EmbeddingError(f"Embedding 服务返回 {e.response.status_code}: {e}") from e
    except httpx.RequestError as e:
        logger.error(f"Embedding API 请求失败: {e}")
        raise EmbeddingError(f"Embedding 服务不可用: {e}") from e
    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"Embedding 响应解析失败: {e}")
        raise EmbeddingError(f"Embedding 响应格式异常: {e}") from e


async def get_embedding(text):
    """单条文本向量生成（批量接口的便捷封装）"""
    return (await get_embeddings([text]))[0]
