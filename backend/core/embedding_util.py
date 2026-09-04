import httpx
import os
from dotenv import load_dotenv
import logging
from backend.config import CONFIG

load_dotenv()
logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """当 Embedding API 调用失败时抛出的自定义异常"""
    pass


async def get_embedding(text):
    """
    异步调用硅基流动 embedding API 获取向量
    :param text: 输入文本
    :return: 向量列表
    :raises EmbeddingError: API 调用失败或响应解析失败时抛出
    :raises ValueError: 输入文本为空时抛出
    """
    if not text or not text.strip():
        raise ValueError("输入文本为空，无法生成向量")

    url = f"{CONFIG.SF_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {CONFIG.SF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": CONFIG.EMBEDDING_MODEL,
        "input": text
    }

    try:
        async with httpx.AsyncClient(timeout=CONFIG.REQUEST_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            # 校验返回结构是否完整（防御性编程）
            if not data.get("data") or not data["data"][0].get("embedding"):
                raise EmbeddingError("API 返回数据结构异常，缺少 embedding 字段")

            return data["data"][0]["embedding"]

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