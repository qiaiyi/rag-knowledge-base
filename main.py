import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header  
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


# ===================== 新增：API Key 鉴权（零信任安全边界） =====================
# 从环境变量读取预设密钥（务必在 .env 中配置 ADMIN_API_KEY）
EXPECTED_API_KEY = os.getenv("ADMIN_API_KEY")

async def validate_api_key(api_key: str = Header(..., alias="X-API-Key")):
    """
    全局鉴权依赖函数：验证请求头中的 X-API-Key
    - 如果密钥匹配 -> 返回密钥本身（供后续使用，此处仅作验证）
    - 如果缺失或不匹配 -> 抛出 403 禁止访问
    """
    if api_key != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=403, 
            detail="无效的 API Key，拒绝访问"
        )
    return api_key
# ============================================================================


# ===== 修改：在创建 app 时挂载全局鉴权依赖 =====
app = FastAPI(dependencies=[Depends(validate_api_key)])

# 允许跨域（Streamlit 前端）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局知识库实例（持久化到 ./chroma_data）
kb = KnowledgeBase("main_kb")

# 允许的文件扩展名和大小限制
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ===== 新增：根路径豁免鉴权（方便健康检查/演示） =====
@app.get("/", dependencies=[])  # dependencies=[] 表示此路径不需要 API Key
def root():
    return {"message": "RAG 知识库 API v2.0 运行中，请携带 X-API-Key 访问其他接口"}


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
    
    return StreamingResponse(stream_answer_with_context(question, chunks), media_type="text/plain")