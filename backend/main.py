import os
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from backend.core.document_loader import load_document
from backend.core.text_splitter import split_text
from backend.core.vector_store import KnowledgeBase
from dotenv import load_dotenv
from backend.core.rag_chain import answer_question, stream_answer_with_context, retrieve_chunks_detailed
from backend.core.http_client import close_client
from backend.storage import ChatStore
from backend.config import CONFIG
from backend.agent.agent import build_react_agent

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

@asynccontextmanager
async def lifespan(app):
    # 应用关闭时释放共享 httpx 连接池
    yield
    await close_client()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 初始化知识库、会话存储和 Agent ==========
kb = KnowledgeBase("main_kb")
store = ChatStore(os.getenv("CHAT_DB_PATH", "./chat_history.db"))
react_agent = build_react_agent(kb)

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
MAX_FILE_SIZE = 10 * 1024 * 1024


def sse_event(event, data):
    """格式化为标准 SSE 帧（event + data 两行，空行结尾）"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ========== 健康检查（无需 API Key） ==========
@app.get("/")
def root():
    return {"message": "RAG 知识库 API v2.0 运行中，请携带 X-API-Key 访问其他接口"}


@app.get("/health")
def health():
    """健康检查端点，用于探针检测"""
    return {"status": "healthy"}


# ========== 文件上传 ==========
@app.post("/upload", dependencies=[Depends(validate_api_key)])
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
        uploaded_at = datetime.now().isoformat(timespec="seconds")
        metadatas = [{"source": file.filename, "uploaded_at": uploaded_at} for _ in chunks]
        await kb.add_documents(chunks, metadatas=metadatas)
        return {"message": f"成功上传 {file.filename}，共添加 {len(chunks)} 个片段"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ========== 文档管理 ==========
@app.get("/documents", dependencies=[Depends(validate_api_key)])
async def list_documents():
    """列出知识库中的所有文档（文件名、片段数、最近上传时间）"""
    return {"documents": await kb.list_documents()}


@app.delete("/documents/{filename}", dependencies=[Depends(validate_api_key)])
async def delete_document(filename: str):
    """删除指定文档的全部向量片段（重新上传同名文档前先删除即可完成更新）"""
    deleted = await kb.delete_document(filename)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"文档 '{filename}' 不存在于知识库中")
    return {"message": f"已删除文档 '{filename}' 的 {deleted} 个片段"}


# ========== 会话管理 ==========
@app.get("/sessions", dependencies=[Depends(validate_api_key)])
def list_sessions():
    return {"sessions": store.list_sessions()}


@app.get("/sessions/{session_id}/messages", dependencies=[Depends(validate_api_key)])
def get_session_messages(session_id: str):
    messages = store.get_messages(session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"session_id": session_id, "messages": messages}


@app.delete("/sessions/{session_id}", dependencies=[Depends(validate_api_key)])
def delete_session(session_id: str):
    if not store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"message": "会话已删除"}


# ========== 普通 RAG 问答（无 Agent，支持多轮会话） ==========
@app.post("/ask", dependencies=[Depends(validate_api_key)])
async def ask(question: str, session_id: str = None):
    if not question or question.strip() == "":
        raise HTTPException(status_code=400, detail="问题不能为空")
    history = []
    if session_id:
        if not store.session_exists(session_id):
            raise HTTPException(status_code=404, detail="会话不存在")
        history = store.get_history(session_id, max_messages=CONFIG.HISTORY_MAX_MESSAGES)

    answer, sources = await answer_question(
        question,
        kb,
        top_k=CONFIG.TOP_K,
        score_threshold=CONFIG.SCORE_THRESHOLD,
        use_rerank=CONFIG.USE_RERANK,
        rerank_top_n=CONFIG.RERANK_TOP_N,
        use_query_rewrite=CONFIG.USE_QUERY_REWRITE,
        history=history
    )
    if not session_id:
        session_id = store.create_session(title=question.strip()[:50])
    store.add_message(session_id, "user", question)
    store.add_message(session_id, "assistant", answer, sources=sources)
    return {"answer": answer, "sources": sources, "session_id": session_id}


# ========== 流式问答（SSE：sources / delta / done 三种事件） ==========
@app.post("/ask/stream", dependencies=[Depends(validate_api_key)])
async def ask_stream(question: str, session_id: str = None):
    if not question or question.strip() == "":
        raise HTTPException(status_code=400, detail="问题不能为空")
    history = []
    if session_id:
        if not store.session_exists(session_id):
            raise HTTPException(status_code=404, detail="会话不存在")
        history = store.get_history(session_id, max_messages=CONFIG.HISTORY_MAX_MESSAGES)

    hits = await retrieve_chunks_detailed(
        question,
        kb,
        top_k=CONFIG.TOP_K,
        score_threshold=CONFIG.SCORE_THRESHOLD,
        use_rerank=CONFIG.USE_RERANK,
        rerank_top_n=CONFIG.RERANK_TOP_N,
        use_query_rewrite=CONFIG.USE_QUERY_REWRITE,
        history=history
    )
    if not session_id:
        session_id = store.create_session(title=question.strip()[:50])
    store.add_message(session_id, "user", question)

    async def event_stream():
        answer_text = ""
        try:
            # 1. 检索/重排完成后先下发来源（含文件名元数据）
            yield sse_event("sources", hits)
            if not hits:
                answer_text = "知识库中暂时没有相关内容。"
                yield sse_event("delta", {"text": answer_text})
            else:
                # 2. 正文逐段推送
                async for delta in stream_answer_with_context(question, [h["content"] for h in hits]):
                    answer_text += delta
                    yield sse_event("delta", {"text": delta})
            # 3. 结束标记（携带会话 id，便于前端绑定新建的会话）
            yield sse_event("done", {"session_id": session_id})
        finally:
            store.add_message(session_id, "assistant", answer_text, sources=hits)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ========== Agentic RAG（ReAct Agent） ==========
class AgentQuery(BaseModel):
    question: str

@app.post("/agent/react", dependencies=[Depends(validate_api_key)])
async def agent_react_endpoint(query: AgentQuery):
    """
    ReAct Agent 端点：
    - 自动判断是否需要调用工具（知识库检索 / 网络搜索 / 计算器）
    - 返回最终答案 + 完整的推理轨迹（trajectory）
    """
    if not query.question or not query.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        config = {"recursion_limit": CONFIG.AGENT_RECURSION_LIMIT}

        result = await react_agent.ainvoke(
            {"messages": [HumanMessage(content=query.question)]},
            config=config
        )

        final_msg = result["messages"][-1]
        answer = final_msg.content or "处理完成，但未生成回答。"

        # 提取推理轨迹（便于调试和前端展示）
        trajectory = []
        for m in result["messages"]:
            row = {"role": m.type, "content": m.content}
            tool_calls = getattr(m, "tool_calls", None) or []
            if tool_calls:
                row["tool_calls"] = [
                    {
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": tc.get("args", ""),
                        },
                    }
                    for tc in tool_calls
                ]
            trajectory.append(row)

        return {
            "question": query.question,
            "answer": answer,
            "trajectory": trajectory
        }
    except Exception as e:
        logging.error(f"ReAct Agent 处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent 处理失败: {str(e)}")
