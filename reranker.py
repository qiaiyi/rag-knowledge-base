import requests
import os
from dotenv import load_dotenv

load_dotenv()

def rerank(query, documents, top_n=3):
    """
    使用硅基流动的 BGE-reranker 对文档片段进行重排序
    :param query: 用户问题
    :param documents: 候选文档列表（一般来自向量检索的 top-k）
    :param top_n: 返回的最相关文档数量
    :return: 重排序后的文档列表（按相关性降序）
    """
    url = f"{os.getenv('SF_BASE_URL')}/rerank"
    headers = {
        "Authorization": f"Bearer {os.getenv('SF_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": query,
        "documents": documents,
        "top_n": top_n   # 直接让 API 返回前 top_n 个
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # 返回结构: {"results": [{"index": 0, "relevance_score": 0.99}, ...]}
        # 按 relevance_score 降序排序
        sorted_results = sorted(data["results"], key=lambda x: x["relevance_score"], reverse=True)
        ranked_docs = []
        for item in sorted_results:
            idx = item["index"]
            ranked_docs.append(documents[idx])
        return ranked_docs[:top_n]
    except Exception as e:
        print(f"Rerank 调用失败: {e}")
        # 降级：返回原来的前 top_n 个
        return documents[:top_n]