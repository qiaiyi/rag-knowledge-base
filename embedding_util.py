import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_embedding(text):
    """调用硅基流动 embedding API 获取向量"""
    url = f"{os.getenv('SF_BASE_URL')}/embeddings"
    headers = {
        "Authorization": f"Bearer {os.getenv('SF_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "BAAI/bge-large-zh-v1.5",
        "input": text
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        print(f"Embedding 调用失败: {e}")
        return None

if __name__ == "__main__":
    vec = get_embedding("人工智能改变了世界")
    if vec:
        print(f"向量长度：{len(vec)}")
        print(f"前5个值：{vec[:5]}")
    else:
        print("获取向量失败，请检查网络和 API Key")