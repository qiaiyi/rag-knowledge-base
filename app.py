import streamlit as st
import requests

# 后端 API 地址（根据实际情况修改）
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="个人知识库问答助手", layout="wide")
st.title("📚 个人知识库问答助手")

# 侧边栏：文件上传
with st.sidebar:
    st.header("上传文档")
    uploaded_file = st.file_uploader("选择文件（PDF / TXT / DOCX）", type=["pdf", "txt", "docx"])
    if uploaded_file is not None:
        # 显示文件信息
        st.info(f"已选择：{uploaded_file.name}")
        if st.button("上传并建立索引"):
            with st.spinner("正在处理文档，请稍候..."):
                # 发送文件到后端
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
            # 调用问答接口
            resp = requests.post(f"{API_URL}/ask", params={"question": question})
            if resp.status_code == 200:
                data = resp.json()
                st.markdown("### 回答")
                st.write(data["answer"])
                st.caption("💡 答案中的【数字】对应下方的参考来源编号")
                
                # 显示引用来源（可折叠）
                if data.get("sources"):
                    with st.expander("📖 参考来源"):
                        for i, src in enumerate(data["sources"], 1):
                            st.markdown(f"**[{i}]** {src[:300]}...")
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