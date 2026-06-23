import os
import json
import logging
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from document_loader import load_document
from text_splitter import split_text
from vector_store import KnowledgeBase
from dotenv import load_dotenv
from rag_chain import answer_question, stream_answer_with_context, retrieve_chunks
from config import CONFIG

load_dotenv()

# 配置全局日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# 抑制第三方库的调试日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


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

# 允许的文件扩展名和大小限制（放在文件顶部常量区）
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文档并建立索引（支持 .txt, .pdf, .docx，文件大小 ≤ 10MB）"""
    
    # 1. 校验文件扩展名
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 '{ext}'，仅允许: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # 2. 读取文件内容并校验大小
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件大小 {len(contents)//1024} KB 超过最大限制 {MAX_FILE_SIZE//(1024*1024)} MB"
        )
    
    # 3. 保存临时文件
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(contents)
        
        # 4. 加载、切分、入库
        text = load_document(temp_path)
        chunks = split_text(text, chunk_size=CONFIG.CHUNK_SIZE, overlap=CONFIG.CHUNK_OVERLAP)
        metadatas = [{"source": file.filename} for _ in chunks]
        # add_documents 现在是异步方法，需要 await
        await kb.add_documents(chunks, metadatas=metadatas)
        
        return {"message": f"成功上传 {file.filename}，共添加 {len(chunks)} 个片段"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/ask")
async def ask(question: str):
    """
    非流式问答接口（使用优化后的 RAG 流程：重排序、查询重写、阈值过滤等）
    """
    if not question or question.strip() == "":
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    # answer_question 现在是异步函数，需要 await
    answer, sources = await answer_question(
        question,
        kb,
        top_k=CONFIG.TOP_K,
        score_threshold=CONFIG.SCORE_THRESHOLD,
        use_rerank=CONFIG.USE_RERANK,
        rerank_top_n=CONFIG.RERANK_TOP_N,
        use_query_rewrite=CONFIG.USE_QUERY_REWRITE
    )
    return {"answer": answer, "sources": sources}


@app.post("/ask/stream")
async def ask_stream(question: str):
    if not question or question.strip() == "":
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    # retrieve_chunks 现在是异步函数，需要 await
    chunks = await retrieve_chunks(
        question,
        kb,
        top_k=CONFIG.TOP_K,
        score_threshold=CONFIG.SCORE_THRESHOLD,
        use_rerank=CONFIG.USE_RERANK,
        rerank_top_n=CONFIG.RERANK_TOP_N,
        use_query_rewrite=CONFIG.USE_QUERY_REWRITE
    )
    
    if not chunks:
        return StreamingResponse(iter(["知识库中暂时没有相关内容。"]), media_type="text/plain")
    
    # stream_answer_with_context 是异步生成器，StreamingResponse 可直接接受
    return StreamingResponse(stream_answer_with_context(question, chunks), media_type="text/plain")


@app.get("/")
def root():
    return {"message": "RAG 知识库 API v2.0 运行中"}