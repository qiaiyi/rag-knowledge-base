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


# ---------- Helpers ----------
def highlight_citations(text: str) -> str:
    repl = r'<span style="background-color: rgba(255, 255, 150, 0.5); padding: 0 2px; border-radius: 4px;">\g<0></span>'
    return re.sub(r'【\d+】', repl, text)


def format_sources(sources: list[str], max_len: int = 300) -> list[str]:
    return [
        f"**[{i}]** {src[:max_len]}..." if len(src) > max_len else f"**[{i}]** {src}"
        for i, src in enumerate(sources, 1)
    ]


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


# ---------- Page ----------
st.set_page_config(page_title="个人知识库问答助手", layout="wide")
st.title("个人知识库问答助手")

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role", "content", "sources", "trajectory"}
if "upload_status" not in st.session_state:
    st.session_state.upload_status = None

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
    if st.session_state.upload_status:
        st.info(st.session_state.upload_status)
    else:
        st.caption("尚未上传任何文档，请上传以建立索引。")

    st.divider()
    st.subheader("控制面板")

    # Agent mode toggle
    use_agent = st.toggle("Agent 模式", value=False,
                          help="开启后使用 ReAct Agent，可自动调用知识库检索、网络搜索、计算器等工具")
    # Stream toggle — only relevant in standard mode
    use_stream = st.toggle("流式输出", value=True,
                           help="开启后答案将逐字显示（标准 RAG 模式）")

    if st.button("清空对话历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

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
            highlighted = highlight_citations(msg.get("content", ""))
            st.markdown(highlighted, unsafe_allow_html=True)

            sources = msg.get("sources", [])
            if sources:
                with st.expander("查看参考来源"):
                    for s in format_sources(sources):
                        st.markdown(s)

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
            # ===== Agent mode =====
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

                    # Extract plain-text snippets from tool results as sources
                    sources = [
                        step["content"][:200]
                        for step in trajectory
                        if step.get("role") == "tool" and step.get("content")
                    ]

                    st.markdown(highlight_citations(answer), unsafe_allow_html=True)
                    if sources:
                        with st.expander("查看参考来源"):
                            for s in format_sources(sources):
                                st.markdown(s)
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
            if use_stream:
                # --- Streaming ---
                try:
                    resp = requests.post(
                        STREAM_ENDPOINT,
                        params={"question": question},
                        headers=HEADERS,
                        stream=True,
                        timeout=120,
                    )
                    resp.raise_for_status()

                    placeholder = st.empty()
                    full_response = ""
                    sources = []

                    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                        if not chunk:
                            continue
                        full_response += chunk
                        if "__SOURCES__:" in full_response:
                            parts = full_response.split("__SOURCES__:", 1)
                            answer_text = parts[0]
                            try:
                                sources = json.loads(parts[1]) if len(parts) > 1 else []
                            except json.JSONDecodeError:
                                sources = []
                            placeholder.markdown(highlight_citations(answer_text),
                                                unsafe_allow_html=True)
                        else:
                            placeholder.markdown(f"{full_response}▌",
                                                unsafe_allow_html=True)

                    if "__SOURCES__:" in full_response:
                        answer_text = full_response.split("__SOURCES__:", 1)[0]
                    else:
                        answer_text = full_response

                    placeholder.markdown(highlight_citations(answer_text),
                                        unsafe_allow_html=True)

                    if sources:
                        with st.expander("查看参考来源"):
                            for s in format_sources(sources):
                                st.markdown(s)

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
                            params={"question": question},
                            headers=HEADERS,
                            timeout=90,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        answer = data.get("answer", "")
                        sources = data.get("sources", [])

                        st.markdown(highlight_citations(answer), unsafe_allow_html=True)
                        if sources:
                            with st.expander("查看参考来源"):
                                for s in format_sources(sources):
                                    st.markdown(s)

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
st.caption("提示：答案中的【数字】对应参考来源编号。Agent 模式下可自动调用知识库检索、网络搜索等工具。")
