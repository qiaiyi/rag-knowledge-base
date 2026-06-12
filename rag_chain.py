import requests
import os
from dotenv import load_dotenv
from vector_store import KnowledgeBase
from document_loader import load_document
from text_splitter import split_text

load_dotenv()

# DeepSeek 聊天配置
LLM_URL = f"{os.getenv('BASE_URL')}/chat/completions"
LLM_KEY = os.getenv("API_KEY")

def ask_with_context(question, context_chunks, model="deepseek-v3-2-251201"):
    # 给每个片段加上编号标签
    numbered_chunks = []
    for i, chunk in enumerate(context_chunks, 1):
        numbered_chunks.append(f"[{i}] {chunk}")
    context_text = "\n\n".join(numbered_chunks)
    
    # 修改 Prompt，要求引用编号
    prompt = f"""你是一个知识助手。请根据以下资料回答问题。如果资料中没有相关信息，请如实说“资料中未提及”，不要编造答案。
在回答中，请在你使用的信息后面用【数字】标注来源，例如【1】。

资料：
{context_text}

问题：{question}
答案："""
    headers = {
        "Authorization": f"Bearer {LLM_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    resp = requests.post(LLM_URL, json=payload, headers=headers)
    return resp.json()["choices"][0]["message"]["content"]

# 完整流程演示
if __name__ == "__main__":
    # 1. 加载文档
    docs = load_document("test.txt")
    # 2. 切分
    chunks = split_text(docs)
    # 3. 建立向量库
    kb = KnowledgeBase("demo_kb")
    metadatas = [{"source": "test.txt"}] * len(chunks)
    kb.add_documents(chunks, metadatas=metadatas)
    # 4. 提问
    query = "如何写出亮眼的网文开头？"
    retrieved = kb.search(query, top_k=2)
    print("检索到的片段：", retrieved)
    # 5. 生成答案
    answer = ask_with_context(query, retrieved)
    print("\n最终答案：", answer)