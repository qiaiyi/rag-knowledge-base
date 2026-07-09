# main.py
import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from document_loader import load_document
from text_splitter import split_text
from vector_store import KnowledgeBase
from dotenv import load_dotenv
from rag_chain import answer_question, stream_answer_with_context, retrieve_chunks
from config import CONFIG
from agent import build_react_agent  # ✅ 改为从 agent 导入

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

EXPECTED_API_KEY = os.getenv("ADMIN_API_KEY")

async def validate_api_key(api_key: str = Header(..., alias="X-API-Key")):
    if api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=403, detail="无效的 API Key，拒绝访问")
    return api_key

app = FastAPI(dependencies=[Depends(validate_api_key)])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 初始化知识库和 Agent ==========
kb = KnowledgeBase("main_kb")
react_agent = build_react_agent(kb)   # ✅ kb 通过闭包注入，无需在 invoke 时再传

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
MAX_FILE_SIZE = 10 * 1024 * 1024


# ========== 健康检查（无需 API Key） ==========
@app.get("/", dependencies=[])
def root():
    return {"message": "RAG 知识库 API v2.0 运行中，请携带 X-API-Key 访问其他接口"}


@app.get("/health", dependencies=[])
def health():
    """健康检查端点，用于探针检测"""
    return {"status": "healthy"}


# ========== 文件上传 ==========
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 '{ext}'，仅允许: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件大小 {len(contents)//1024} KB 超过最大限制 {MAX_FILE_SIZE//(1024*1024)} MB"
        )
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(contents)
        text = load_document(temp_path)
        chunks = split_text(text, chunk_size=CONFIG.CHUNK_SIZE, overlap=CONFIG.CHUNK_OVERLAP, method=CONFIG.CHUNK_METHOD)
        metadatas = [{"source": file.filename} for _ in chunks]
        await kb.add_documents(chunks, metadatas=metadatas)
        return {"message": f"成功上传 {file.filename}，共添加 {len(chunks)} 个片段"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ========== 普通 RAG 问答（无 Agent） ==========
@app.post("/ask")
async def ask(question: str):
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


# ========== Agentic RAG（ReAct Agent） ==========
class AgentQuery(BaseModel):
    question: str


@app.post("/agent/react")
async def agent_react_endpoint(query: AgentQuery):
    """
    ReAct Agent 端点：
    - 自动判断是否需要调用工具（知识库检索 / 网络搜索 / 计算器）
    - 返回最终答案 + 完整的推理轨迹（trajectory）
    """
    if not query.question or not query.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    try:
        # ✅ 设置 recursion_limit，防止死循环
        config = {"recursion_limit": CONFIG.AGENT_RECURSION_LIMIT}
        
        result = await react_agent.ainvoke(
            {"messages": [{"role": "user", "content": query.question}]},
            config=config  # 传入限制
        )
        
        final_msg = result["messages"][-1]
        answer = final_msg.get("content", "处理完成，但未生成回答。")
        
        # 提取推理轨迹（便于调试和前端展示）
        trajectory = [
            {
                "role": m.get("role"),
                "content": m.get("content", ""),
                "tool_calls": m.get("tool_calls")
            }
            for m in result["messages"]
        ]
        
        return {
            "question": query.question,
            "answer": answer,
            "trajectory": trajectory
        }
    except Exception as e:
        logging.error(f"ReAct Agent 处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent 处理失败: {str(e)}")
