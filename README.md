# 📚 RAG 个人知识库问答助手

基于 RAG（检索增强生成）技术的智能问答系统，支持上传 PDF/TXT/DOCX 文档，通过语义检索和大模型生成回答，并标注引用来源。

## ✨ 功能特点

- 支持多种文档格式（PDF、TXT、DOCX）
- 智能文本切分（可配置块大小与重叠）
- 使用 BAAI/bge-m3 嵌入模型 + Chroma 向量数据库
- 基于 DeepSeek 大模型生成答案，带引用编号
- 提供 FastAPI 后端 + Streamlit 前端界面
- 流式输出（可选）提升用户体验

## 🛠️ 技术栈

- **后端**：FastAPI, Chroma, requests
- **前端**：Streamlit
- **嵌入模型**：硅基流动 BAAI/bge-m3（免费）
- **大模型**：DeepSeek API / 硅基流动（可配置）
- **文档处理**：pypdf, python-docx

