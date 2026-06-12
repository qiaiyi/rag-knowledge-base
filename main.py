from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import shutil
import os
import json
import requests
from document_loader import load_document
from text_splitter import split_text
from vector_store import KnowledgeBase
from dotenv import load_dotenv

load_dotenv()

# DeepSeek API 配置
LLM_URL = f"{os.getenv('BASE_URL')}/chat/completions"
LLM_KEY = os.getenv("API_KEY")

app = FastAPI()

# 允许跨域（Streamlit 前端）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局知识库实例（持久化到 ./chroma_data）
kb = KnowledgeBase("main_kb")

# ------------------- 非流式问答（带引用） -------------------
def ask_with_context(question, context_chunks, model="deepseek-v3-2-251201"):
    """根据检索到的文档片段生成答案，并在 Prompt 中要求添加引用编号"""
    # 给每个片段加上编号标签 [1], [2], ...
    numbered_chunks = [f"[{i}] {chunk}" for i, chunk in enumerate(context_chunks, 1)]
    context_text = "\n\n".join(numbered_chunks)
    
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
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    try:
        resp = requests.post(LLM_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return "抱歉，生成答案时遇到错误。"

# ------------------- 流式生成器 -------------------
def stream_llm_answer(question, context_chunks, model="deepseek-v3-2-251201"):
    """生成器，逐步产生答案的字符片段（用于流式输出）"""
    numbered_chunks = [f"[{i}] {chunk}" for i, chunk in enumerate(context_chunks, 1)]
    context_text = "\n\n".join(numbered_chunks)
    
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
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "stream": True   # 开启流式
    }
    try:
        with requests.post(LLM_URL, json=payload, headers=headers, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                            delta = chunk_data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield f"流式生成出错: {e}"

# ------------------- API 接口 -------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文档并建立索引"""
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        text = load_document(temp_path)
        chunks = split_text(text, chunk_size=500, overlap=50)
        metadatas = [{"source": file.filename}] * len(chunks)
        kb.add_documents(chunks, metadatas=metadatas)
        return {"message": f"成功上传 {file.filename}，共添加 {len(chunks)} 个片段"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/ask")
async def ask(question: str):
    """非流式问答接口（返回完整答案和来源）"""
    if not question or question.strip() == "":
        raise HTTPException(status_code=400, detail="问题不能为空")
    chunks = kb.search(question, top_k=3)
    if not chunks:
        return {"answer": "知识库中暂时没有相关内容。", "sources": []}
    answer = ask_with_context(question, chunks)
    return {"answer": answer, "sources": chunks}

@app.post("/ask/stream")
async def ask_stream(question: str):
    """流式问答接口（逐字返回答案）"""
    if not question or question.strip() == "":
        raise HTTPException(status_code=400, detail="问题不能为空")
    chunks = kb.search(question, top_k=3)
    if not chunks:
        # 如果没有检索结果，直接返回错误信息（非流式）
        return {"answer": "知识库中暂时没有相关内容。", "sources": []}
    return StreamingResponse(stream_llm_answer(question, chunks), media_type="text/plain")

@app.get("/")
def root():
    return {"message": "RAG 知识库 API 正在运行"}