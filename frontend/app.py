import streamlit as st
import requests
import re
import os
import json
from dotenv import load_dotenv

load_dotenv()

# ---------- Configuration ----------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("ADMIN_API_KEY", "")
if not API_KEY:
    st.error("未设置 ADMIN_API_KEY 环境变量，请在 .env 中配置")
    st.stop()

HEADERS = {"X-API-Key": API_KEY}

UPLOAD_ENDPOINT = f"{API_URL}/upload"
NONSTREAM_ENDPOINT = f"{API_URL}/ask"
STREAM_ENDPOINT = f"{API_URL}/ask/stream"
AGENT_ENDPOINT = f"{API_URL}/agent/react"
SESSIONS_ENDPOINT = f"{API_URL}/sessions"


# ---------- Helpers ----------
def highlight_citations(text: str) -> str:
    repl = r'<span style="background-color: rgba(255, 255, 150, 0.5); padding: 0 2px; border-radius: 4px;">\g<0></span>'
    return re.sub(r'【\d+】', repl, text)


def source_text(src) -> str:
    """来源项可能是字符串（Agent 模式）或 {"content","source"} 字典（RAG 模式）"""
    if isinstance(src, dict):
        return src.get("content", "")
    return src


def source_origin(src) -> str | None:
    return src.get("source") if isinstance(src, dict) else None


def format_sources(sources: list, max_len: int = 300) -> list[str]:
    items = []
    for i, src in enumerate(sources, 1):
        text = source_text(src)
        short = text[:max_len] + ("..." if len(text) > max_len else "")
        origin = source_origin(src)
        line = f"**[{i}]** {short}"
        if origin:
            line += f"\n\n- 📄 文件：`{origin}`"
        items.append(line)
    return items


def parse_sse_stream(resp):
    """
    解析标准 SSE 流（event: xxx / data: {json} 帧，空行分隔）
    逐帧 yield (event, data)
    """
    buffer = ""
    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
        if not chunk:
            continue
        buffer += chunk
        while "\n\n" in buffer:
            raw, buffer = buffer.split("\n\n", 1)
            event, data_str = None, None
            for line in raw.split("\n"):
                if line.startswith("event: "):
                    event = line[len("event: "):].strip()
                elif line.startswith("data: "):
                    data_str = line[len("data: "):]
            if event is None or data_str is None:
                continue
            try:
                yield event, json.loads(data_str)
            except json.JSONDecodeError:
                continue


def render_trajectory(trajectory: list[dict]):
    """Render agent reasoning trajectory inside an expander."""
    with st.expander("🧠 查看 Agent 推理轨迹"):
        for step in trajectory:
            role = step.get("role", "unknown")
            content = step.get("content", "")
            tool_calls = step.get("tool_calls")
            if role == "user":
                st.markdown(f"**👤 用户:** {content}")
            elif role == "assistant" and tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    st.markdown(f"**🔧 调用工具:** `{fn.get('name', 'unknown')}`")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                        st.code(json.dumps(args, ensure_ascii=False, indent=2), language="json")
                    except json.JSONDecodeError:
                        st.code(fn.get("arguments", ""), language="json")
            elif role == "tool":
                st.markdown(f"**📊 工具结果:**")
                st.text(content[:500] + ("..." if len(content) > 500 else ""))
            elif role == "assistant":
                st.markdown(f"**🤖 Agent:** {content}")
            st.divider()


def fetch_sessions() -> list[dict]:
    try:
        resp = requests.get(SESSIONS_ENDPOINT, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json().get("sessions", [])
    except Exception:
        return []


def render_answer(answer: str, sources: list):
    st.markdown(highlight_citations(answer), unsafe_allow_html=True)
    if sources:
        with st.expander("查看参考来源"):
            for s in format_sources(sources):
                st.markdown(s)


# ---------- Page ----------
st.set_page_config(page_title="个人知识库问答助手", layout="wide")
st.title("个人知识库问答助手")

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role", "content", "sources", "trajectory"}
if "upload_status" not in st.session_state:
    st.session_state.upload_status = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# ---------- Sidebar ----------
with st.sidebar:
    st.header("文档管理")
    uploaded_file = st.file_uploader("上传文档（PDF / TXT / DOCX）", type=["pdf", "txt", "docx"])
    if uploaded_file is not None:
        st.info(f"已选择：{uploaded_file.name}")
        if st.button("上传并建立索引", use_container_width=True):
            with st.spinner("正在处理文档，请稍候..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    resp = requests.post(UPLOAD_ENDPOINT, files=files, headers=HEADERS, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        msg = data.get("message", "")
                        st.session_state.upload_status = f"成功：{msg}"
                        st.success(st.session_state.upload_status)
                        st.rerun()
                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"上传失败：{detail}")
                except requests.Timeout:
                    st.error("上传超时，请检查网络或文件大小")
                except requests.ConnectionError:
                    st.error("无法连接到后端服务，请确保 API 已启动")
                except Exception as e:
                    st.error(f"发生未知错误：{e}")

    st.divider()
    st.subheader("知识库状态")
    try:
        docs_resp = requests.get(f"{API_URL}/documents", headers=HEADERS, timeout=10)
        if docs_resp.status_code == 200:
            docs = docs_resp.json().get("documents", [])
            if docs:
                for doc in docs:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"📄 {doc['source']}（{doc['chunks']} 片段）")
                    with col2:
                        if st.button("删除", key=f"del_{doc['source']}", help="删除该文档全部片段"):
                            del_resp = requests.delete(
                                f"{API_URL}/documents/{doc['source']}",
                                headers=HEADERS, timeout=30,
                            )
                            if del_resp.status_code == 200:
                                st.toast("已删除")
                                st.rerun()
                            else:
                                st.error(del_resp.json().get("detail", "删除失败"))
            else:
                st.caption("知识库为空，请上传文档建立索引。")
        elif st.session_state.upload_status:
            st.info(st.session_state.upload_status)
    except Exception:
        st.caption("后端未启动，无法获取知识库状态。")

    st.divider()
    st.subheader("对话历史")
    if st.button("➕ 新建对话", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

    sessions = fetch_sessions()
    if sessions:
        options = {f"{s['title'] or '（无标题）'} · {s['created_at']}（{s['message_count']} 条）": s["id"]
                   for s in sessions}
        selected = st.selectbox("历史会话", list(options.keys()), index=None)
        if selected and st.button("载入此会话", use_container_width=True):
            try:
                resp = requests.get(f"{SESSIONS_ENDPOINT}/{options[selected]}/messages",
                                    headers=HEADERS, timeout=10)
                if resp.status_code == 200:
                    msgs = resp.json().get("messages", [])
                    st.session_state.messages = [
                        {"role": m["role"], "content": m["content"], "sources": m.get("sources", [])}
                        for m in msgs
                    ]
                    st.session_state.session_id = options[selected]
                    st.rerun()
                else:
                    st.error("会话不存在或已被删除")
            except Exception as e:
                st.error(f"载入失败：{e}")
        if selected and st.button("🗑️ 删除此会话", use_container_width=True):
            try:
                resp = requests.delete(f"{SESSIONS_ENDPOINT}/{options[selected]}",
                                       headers=HEADERS, timeout=10)
                if resp.status_code == 200:
                    if st.session_state.session_id == options[selected]:
                        st.session_state.session_id = None
                        st.session_state.messages = []
                    st.rerun()
                else:
                    st.error("删除失败")
            except Exception as e:
                st.error(f"删除失败：{e}")
    else:
        st.caption("暂无历史会话")

    st.divider()
    st.subheader("控制面板")

    # Agent mode toggle
    use_agent = st.toggle("Agent 模式", value=False,
                          help="开启后使用 ReAct Agent，可自动调用知识库检索、网络搜索、计算器等工具")
    # Stream toggle — only relevant in standard mode
    use_stream = st.toggle("流式输出", value=True,
                           help="开启后答案将逐字显示（标准 RAG 模式）")

# ---------- Main chat area ----------
st.header("智能问答")
mode_label = "Agent 模式" if use_agent else "标准 RAG 模式"
st.caption(f"当前模式：{mode_label}")

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_answer(msg.get("content", ""), msg.get("sources", []))
            trajectory = msg.get("trajectory", [])
            if trajectory:
                render_trajectory(trajectory)

# ---------- Handle new question ----------
question = st.chat_input("输入你的问题")

if question:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if use_agent:
            # ===== Agent mode（不参与多轮会话） =====
            with st.spinner("Agent 正在思考并调用工具..."):
                try:
                    resp = requests.post(
                        AGENT_ENDPOINT,
                        json={"question": question},
                        headers=HEADERS,
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    answer = data.get("answer", "")
                    trajectory = data.get("trajectory", [])

                    sources = [
                        step["content"][:200]
                        for step in trajectory
                        if step.get("role") == "tool" and step.get("content")
                    ]

                    render_answer(answer, sources)
                    if trajectory:
                        render_trajectory(trajectory)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "trajectory": trajectory,
                    })

                except requests.Timeout:
                    st.error("Agent 请求超时，请稍后重试")
                except requests.ConnectionError:
                    st.error("无法连接到后端服务")
                except Exception as e:
                    st.error(f"发生错误：{e}")
        else:
            # ===== Standard RAG mode =====
            params = {"question": question}
            if st.session_state.session_id:
                params["session_id"] = st.session_state.session_id

            if use_stream:
                # --- Streaming（解析真 SSE 事件流） ---
                try:
                    resp = requests.post(
                        STREAM_ENDPOINT,
                        params=params,
                        headers=HEADERS,
                        stream=True,
                        timeout=120,
                    )
                    resp.raise_for_status()

                    placeholder = st.empty()
                    answer_text = ""
                    sources = []

                    for event, data in parse_sse_stream(resp):
                        if event == "sources":
                            sources = data
                        elif event == "delta":
                            answer_text += data.get("text", "")
                            placeholder.markdown(
                                highlight_citations(answer_text) + "▌",
                                unsafe_allow_html=True,
                            )
                        elif event == "done":
                            if data.get("session_id"):
                                st.session_state.session_id = data["session_id"]

                    render_answer(answer_text, sources)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "trajectory": [],
                    })

                except requests.Timeout:
                    st.error("生成超时，请稍后重试")
                except requests.ConnectionError:
                    st.error("无法连接到后端服务")
                except Exception as e:
                    st.error(f"发生错误：{e}")
            else:
                # --- Non-streaming ---
                with st.spinner("正在生成答案..."):
                    try:
                        resp = requests.post(
                            NONSTREAM_ENDPOINT,
                            params=params,
                            headers=HEADERS,
                            timeout=90,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        answer = data.get("answer", "")
                        sources = data.get("sources", [])
                        if data.get("session_id"):
                            st.session_state.session_id = data["session_id"]

                        render_answer(answer, sources)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "trajectory": [],
                        })

                    except requests.Timeout:
                        st.error("请求超时，请稍后重试")
                    except requests.ConnectionError:
                        st.error("无法连接到后端服务")
                    except Exception as e:
                        st.error(f"发生错误：{e}")

    st.rerun()

st.divider()
st.caption("提示：答案中的【数字】对应参考来源编号。标准模式支持多轮追问（如“它的作用是什么”），会话自动保存，可在左侧切换。Agent 模式下可自动调用知识库检索、网络搜索等工具。")
