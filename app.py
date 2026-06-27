import streamlit as st
import requests
import re
import os
import json
from dotenv import load_dotenv

# 加载 .env 文件（若存在）
load_dotenv()

# ---------- 配置 ----------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
# 读取后端鉴权密钥，若未设置则提示
API_KEY = os.getenv("ADMIN_API_KEY", "")
if not API_KEY:
    st.error("❌ 未设置 ADMIN_API_KEY 环境变量，请在 .env 中配置或系统环境中导出")
    st.stop()

STREAM_ENDPOINT = f"{API_URL}/ask/stream"
UPLOAD_ENDPOINT = f"{API_URL}/upload"
NONSTREAM_ENDPOINT = f"{API_URL}/ask"

# 通用的请求头（携带鉴权）
HEADERS = {"X-API-Key": API_KEY}

# ---------- 辅助函数 ----------
def highlight_citations(text):
    pattern = r'【(\d+)】'
    repl = r'<span style="background-color: rgba(255, 255, 150, 0.5); padding: 0 2px; border-radius: 4px;">【\1】</span>'
    return re.sub(pattern, repl, text)

def format_sources(sources, max_len=300):
    formatted = []
    for i, src in enumerate(sources, 1):
        if len(src) > max_len:
            display = src[:max_len] + "... (点击查看完整)"
        else:
            display = src
        formatted.append(f"**[{i}]** {display}")
    return formatted

# ---------- 页面配置 ----------
st.set_page_config(page_title="个人知识库问答助手", layout="wide")
st.title("📚 个人知识库问答助手")

# ---------- 初始化 session_state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "upload_status" not in st.session_state:
    st.session_state.upload_status = None

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("📄 文档管理")
    uploaded_file = st.file_uploader("上传文档（PDF / TXT / DOCX）", type=["pdf", "txt", "docx"])
    if uploaded_file is not None:
        st.info(f"已选择：{uploaded_file.name}")
        if st.button("🚀 上传并建立索引", use_container_width=True):
            with st.spinner("正在处理文档，请稍候..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    resp = requests.post(UPLOAD_ENDPOINT, files=files, headers=HEADERS, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        msg = data.get("message", "")
                        st.session_state.upload_status = f"✅ {msg}"
                        st.success(st.session_state.upload_status)
                        st.rerun()
                    else:
                        error_detail = resp.json().get("detail", resp.text)
                        st.error(f"上传失败：{error_detail}")
                except requests.Timeout:
                    st.error("⏱️ 上传超时，请检查网络或文件大小")
                except requests.ConnectionError:
                    st.error("❌ 无法连接到后端服务，请确保 API 已启动")
                except Exception as e:
                    st.error(f"发生未知错误：{e}")

    st.divider()
    st.subheader("📊 知识库状态")
    if st.session_state.upload_status:
        st.info(st.session_state.upload_status)
    else:
        st.caption("尚未上传任何文档，请上传以建立索引。")

    st.divider()
    st.subheader("⚙️ 控制面板")
    
    if st.button("🛑 停止生成（结束当前请求）", use_container_width=True):
        st.stop()
    
    if st.button("🗑️ 清空对话历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- 主区域：对话 ----------
st.header("💬 智能问答")

for idx, (q, a, srcs) in enumerate(st.session_state.messages):
    with st.chat_message("user"):
        st.markdown(f"**问题 {idx+1}:** {q}")
    with st.chat_message("assistant"):
        st.markdown("**回答:**")
        highlighted = highlight_citations(a)
        st.markdown(highlighted, unsafe_allow_html=True)
        if srcs:
            with st.expander("📖 查看参考来源"):
                for s in format_sources(srcs):
                    st.markdown(s)

question = st.chat_input("输入你的问题，例如：RAG 技术有哪些优点？")
col_ask, _ = st.columns([1, 5])
with col_ask:
    use_stream = st.toggle("流式输出", value=True, help="开启后答案将逐字显示，体验更佳")

if question:
    if use_stream:
        try:
            with st.chat_message("user"):
                st.markdown(f"**问题:** {question}")
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                sources = []

                try:
                    resp = requests.post(
                        STREAM_ENDPOINT,
                        params={"question": question},
                        headers=HEADERS,
                        stream=True,
                        timeout=120
                    )
                    resp.raise_for_status()

                    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            if "__SOURCES__:" in full_response:
                                parts = full_response.split("__SOURCES__:", 1)
                                answer_text = parts[0]
                                sources_json = parts[1] if len(parts) > 1 else "[]"
                                try:
                                    sources = json.loads(sources_json)
                                except json.JSONDecodeError:
                                    sources = []
                                highlighted = highlight_citations(answer_text)
                                placeholder.markdown(highlighted, unsafe_allow_html=True)
                            else:
                                placeholder.markdown(f"**回答:**\n\n{full_response}▌", unsafe_allow_html=True)

                    if not sources and "__SOURCES__:" not in full_response:
                        answer_text = full_response
                        sources = []
                        placeholder.markdown(highlight_citations(answer_text), unsafe_allow_html=True)
                    elif sources and "__SOURCES__:" in full_response:
                        answer_text = full_response.split("__SOURCES__:", 1)[0]
                        placeholder.markdown(highlight_citations(answer_text), unsafe_allow_html=True)

                except requests.Timeout:
                    st.error("⏱️ 生成超时，请稍后重试或关闭流式开关")
                except requests.ConnectionError:
                    st.error("❌ 无法连接到后端服务")
                except Exception as e:
                    st.error(f"发生错误：{e}")

                if 'answer_text' not in locals():
                    answer_text = full_response.split("__SOURCES__:", 1)[0] if "__SOURCES__:" in full_response else full_response
                st.session_state.messages.append((question, answer_text, sources))
        except Exception as e:
            st.error(f"流式请求失败：{e}")
    else:
        try:
            with st.chat_message("user"):
                st.markdown(f"**问题:** {question}")
            with st.chat_message("assistant"):
                with st.spinner("正在生成答案..."):
                    resp = requests.post(
                        NONSTREAM_ENDPOINT,
                        params={"question": question},
                        headers=HEADERS,
                        timeout=90
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    highlighted = highlight_citations(answer)
                    st.markdown(highlighted, unsafe_allow_html=True)
                    if sources:
                        with st.expander("📖 参考来源"):
                            for s in format_sources(sources):
                                st.markdown(s)
                st.session_state.messages.append((question, answer, sources))
        except requests.Timeout:
            st.error("⏱️ 请求超时，请稍后重试")
        except requests.ConnectionError:
            st.error("❌ 无法连接到后端服务")
        except Exception as e:
            st.error(f"发生错误：{e}")

st.divider()
st.caption("💡 提示：答案中的【数字】对应参考来源编号。开启流式输出可获得逐字显示效果。")