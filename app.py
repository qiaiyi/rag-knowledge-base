import streamlit as st
import requests
import re

# 后端 API 地址
API_URL = "http://127.0.0.1:8000"

# 高亮答案中的 【数字】
def highlight_citations(text):
    """将【数字】替换为黄色背景的 HTML 标签"""
    pattern = r'【(\d+)】'
    # 使用 span 标签加背景色
    repl = r'<span style="background-color: rgba(255, 255, 150, 0.5); padding: 0 2px; border-radius: 4px;">【\1】</span>'
    return re.sub(pattern, repl, text)

st.set_page_config(page_title="个人知识库问答助手", layout="wide")
st.title("📚 个人知识库问答助手")

# 侧边栏：文件上传
with st.sidebar:
    st.header("上传文档")
    uploaded_file = st.file_uploader("选择文件（PDF / TXT / DOCX）", type=["pdf", "txt", "docx"])
    if uploaded_file is not None:
        st.info(f"已选择：{uploaded_file.name}")
        if st.button("上传并建立索引"):
            with st.spinner("正在处理文档，请稍候..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    resp = requests.post(f"{API_URL}/upload", files=files)
                    if resp.status_code == 200:
                        st.success(resp.json()["message"])
                    else:
                        st.error(f"上传失败：{resp.text}")
                except Exception as e:
                    st.error(f"连接后端失败：{e}")

# 主区域：问答
st.header("💬 提问")
question = st.text_input("输入你的问题", placeholder="例如：RAG 技术有哪些优点？")
col1, col2 = st.columns([1, 5])
with col1:
    ask_button = st.button("提问", type="primary")

if ask_button and question:
    with st.spinner("正在检索并生成答案..."):
        try:
            resp = requests.post(f"{API_URL}/ask", params={"question": question})
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "")
                sources = data.get("sources", [])
                
                # 显示高亮后的答案
                st.markdown("### 回答")
                highlighted_answer = highlight_citations(answer)
                st.markdown(highlighted_answer, unsafe_allow_html=True)
                st.caption("💡 答案中的【数字】对应下方的参考来源编号")
                
                # 显示引用来源（可折叠）
                if sources:
                    with st.expander("📖 参考来源（点击展开）"):
                        for i, src in enumerate(sources, 1):
                            # 截断过长的来源
                            display_src = src[:500] + "..." if len(src) > 500 else src
                            st.markdown(f"**[{i}]** {display_src}")
                else:
                    st.info("没有找到相关参考资料。")
            else:
                st.error(f"请求失败：{resp.text}")
        except Exception as e:
            st.error(f"连接后端失败：{e}")
elif ask_button and not question:
    st.warning("请输入问题")

# 页脚说明
st.markdown("---")
st.caption("基于 RAG 的个人知识库助手 | 使用 DeepSeek + Chroma")