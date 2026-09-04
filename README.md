# RAG 个人知识库问答助手

[![CI](https://github.com/qiaiyi/rag-knowledge-base/actions/workflows/ci.yml/badge.svg)](https://github.com/qiaiyi/rag-knowledge-base/actions/workflows/ci.yml)

基于 RAG（检索增强生成）技术的智能问答系统。支持上传 PDF / DOCX / TXT 文档，自动建立向量索引；内置查询改写、重排序、Agent 推理等进阶能力。

---

## 目录

-  [项目介绍](#项目介绍)
-  [系统架构](#系统架构)
-  [技术栈](#技术栈)
-  [快速开始](#快速开始)
-  [API 接口](#api-接口)
-  [项目文件说明](#项目文件说明)
-  [核心设计要点](#核心设计要点)
-  [项目演进历史](#项目演进历史)
-  [面试准备](#面试准备)

---

## 项目介绍

一个为面试场景打造的 AI 应用开发作品，包含两个层次的问答能力：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **标准 RAG** | 文档 -> 切分 -> 向量化 -> 检索 -> 重排序 -> LLM 生成（带引用） | 基于知识库的精准问答 |
| **Agent 模式** | 使用 LangGraph 搭建 ReAct Agent，可自主调用知识库检索、联网搜索、数学计算等工具 | 需要综合多个信息来源的问题 |

### 亮点特性

- **多策略文本切分**：固定长度 / 按句子 / 按段落三种方式
- **两阶段检索**：向量检索粗召回 -> BGE-reranker 精排
- **查询改写**：让 LLM 将口语化问题转为规范的检索查询
- **相似度阈值过滤**：自动丢弃低相关片段
- **强制引用与防幻觉**：答案标注来源【数字】，无相关信息时明确拒答
- **流式输出**：标准 SSE 事件流（sources/delta/done），前端打字机效果
- **多轮对话**：会话持久化（SQLite），查询改写自动消解指代追问
- **文档管理**：知识库文档列表与删除，支持"先删后传"更新文档
- **Agent 推理轨迹可视化**：在界面上展示 Agent 的思考与工具调用过程
- **自动评估**：LLM-as-Judge 对系统回答打分，生成评估报告

---

## 系统架构

```
用户输入
    |
    v
frontend/app.py (Streamlit 前端)
    |  HTTP 请求 (X-API-Key 鉴权)
    v
backend/main.py (FastAPI 后端)
    |
    +-- /ask, /ask/stream ------> backend/core/rag_chain.py (RAG 核心流程)
    |       |                        |
    |       +-> rewrite_query()       LLM 改写查询（结合多轮历史消解指代）
    |       +-> search_with_scores()  Chroma 向量检索
    |       +-> rerank()              BGE 精排
    |       +-> ask_with_context()    LLM 生成带引用的答案
    |
    +-- /agent/react ------------> backend/agent/agent.py (LangGraph ReAct Agent)
    |       |                        |
    |       +-> agent_node           调 LLM，判断是否要调用工具
    |       +-> tool_node            执行工具（知识库/搜索/计算器）
    |       +-> should_continue      判断是继续还是结束
    |
    +-- /upload -----------------> backend/core/document_loader.py + vector_store.py
    |
    +-- /sessions, /documents ---> backend/storage.py (SQLite) + vector_store.py
```

### 目录结构

```
rag-knowledge-base/
├── backend/                  # 后端应用包
│   ├── main.py               # FastAPI 入口（路由、SSE、鉴权）
│   ├── config.py             # 配置（dataclass + 环境变量）
│   ├── storage.py            # SQLite 会话持久化
│   ├── core/                 # RAG 核心链路（加载→切分→向量化→检索→重排→生成）
│   └── agent/                # LangGraph ReAct Agent 与工具
├── frontend/
│   └── app.py                # Streamlit 界面
├── tests/                    # pytest 测试（67 个）
├── scripts/                  # 评估与调试脚本
├── data/                     # 本地文档（不入库）
└── .github/workflows/ci.yml  # CI
```

---

## 技术栈

### 后端

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI + uvicorn | 异步高性能 |
| 向量数据库 | ChromaDB (PersistentClient) | 持久化到磁盘 |
| 大模型 API | DeepSeek API / 硅基流动 | 可通过环境变量切换 |
| 嵌入模型 | BAAI/bge-large-zh-v1.5 (硅基流动) | 文本向量化 |
| 重排序模型 | BAAI/bge-reranker-v2-m3 (硅基流动) | 精排 |
| 网络搜索 | Tavily API | Agent 联网搜索 |
| Agent 框架 | LangGraph | StateGraph 状态机 |
| 客户端 | httpx (异步) + tenacity (重试) | 网络请求与容错 |

### 前端

| 组件 | 说明 |
|------|------|
| Streamlit | 快速构建交互界面 |
| Requests (同步) | 后端 API 调用 |
| SSE 流式渲染 | 打字机效果 |

---

## 快速开始

### 环境要求

- Python 3.11+
- 网络环境 （需要访问 DeepSeek API 和硅基流动 API）

### 安装

```bash
# 1. 克隆
git clone <repo-url> && cd rag-knowledge-base

# 2. 创建虚拟环境
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 3. 安装依赖（推荐使用精简依赖清单，requirements.txt 为全量快照）
pip install -r requirements-ci.txt

# 4. 配置环境变量
cp .env.example .env   # 然后在 .env 中填入真实密钥
```

### 启动

```bash
# 终端 1：启动后端
uvicorn backend.main:app --reload

# 终端 2：启动前端
streamlit run frontend/app.py
```

访问 `http://localhost:8501` 即可使用。

### 运行测试

```bash
pytest -v
```

测试使用内存版向量库和占位密钥，不访问外部 API、不落盘。CI 会在每次 push / PR 时自动运行（见 `.github/workflows/ci.yml`）。

---

## API 接口

所有接口（除 `/` 和 `/health`）需要在请求头携带 `X-API-Key: <ADMIN_API_KEY>`。

| 路由 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 根路径，健康检查（免鉴权） |
| `/health` | GET | 健康检查（免鉴权） |
| `/upload` | POST | 上传文档（PDF/TXT/DOCX），自动解析并建索引 |
| `/documents` | GET | 列出知识库文档（文件名、片段数、上传时间） |
| `/documents/{filename}` | DELETE | 删除指定文档的全部向量片段（重新上传前先删除即可更新文档） |
| `/ask` | POST | 标准 RAG 问答（非流式），支持 `session_id` 多轮对话 |
| `/ask/stream` | POST | 标准 RAG 问答，**标准 SSE 流**（`sources`/`delta`/`done` 三种事件） |
| `/sessions` | GET | 列出历史会话 |
| `/sessions/{id}/messages` | GET | 查看会话消息历史 |
| `/sessions/{id}` | DELETE | 删除会话 |
| `/agent/react` | POST | Agent 模式问答，返回 `{answer, trajectory}` |

### 流式协议（SSE）

`/ask/stream` 输出标准 `text/event-stream`，三种事件按序推送：

```
event: sources   → 检索/重排完成后先下发引用（含 content/score/source 文件名元数据）
event: delta     → 正文增量（{"text": "..."}），可多次
event: done      → 结束标记（{"session_id": "..."}，新建会话时返回给前端绑定）
```

### 多轮对话

`/ask` 与 `/ask/stream` 接受可选 `session_id` 参数：传入时结合该会话最近 `HISTORY_MAX_MESSAGES` 条消息做**查询改写**（自动消解"它的作用是什么"这类指代追问）；不传时自动新建会话并在响应/`done` 事件中返回。会话与消息持久化在 SQLite（`CHAT_DB_PATH`，默认 `./chat_history.db`）。

---

## 项目文件说明

参考 [项目精读手册.docx](./项目精读手册.docx)（强烈推荐先读此文档），下面只做简要索引：

### 配置文件

- **`.env`** ：API 密钥与运行时配置（不在 git 中）
- **`.gitignore`** ：忽略 venv、.env、chroma_data、csv 等
- **`requirements.txt`** ：Python 依赖

### 核心代码

| 文件 | 职责 | 关键知识点 |
|------|------|-----------|
| `backend/config.py` | 所有配置参数集中管理（dataclass + 环境变量） | dataclass、os.getenv |
| `backend/core/document_loader.py` | 解析 PDF/TXT/DOCX，提取文本 | pypdf、python-docx、边界处理 |
| `backend/core/text_splitter.py` | 按固定长度 / 句子 / 段落切分文本 | chunk 策略、overlap 设计 |
| `backend/core/embedding_util.py` | 调硅基流动 API 生成向量 | httpx 异步请求、异常封装 |
| `backend/core/vector_store.py` | Chroma 封装（增删查 + 去重 + 阈值过滤 + 文档管理） | 向量检索、余弦距离、asyncio.to_thread |
| `backend/core/reranker.py` | BGE-reranker 精排 + 优雅降级 | 两阶段检索模式 |
| `backend/core/rag_chain.py` | RAG 核心流程编排 + 分类重试策略 | 查询改写（支持多轮历史）、tenacity、流式输出 |
| `backend/storage.py` | SQLite 会话/消息持久化（多轮对话） | 标准库 sqlite3、会话 CRUD |
| `backend/agent/tools.py` | Agent 工具（知识库、搜索、计算器）+ Function Schema | AST 安全求值、Function Calling |
| `backend/agent/agent.py` | LangGraph ReAct Agent | 状态机 Node/Edge、reducer |
| `backend/main.py` | FastAPI 后端服务（路由 + 鉴权 + SSE） | 依赖注入、异步路由 |
| `frontend/app.py` | Streamlit 前端（上传 + 对话 + 会话 + Agent 可视化） | session_state、SSE 解析 |
| `scripts/inspect_db.py` | 调试工具：查看知识库内容 | Chroma get() |

### 评估与工具

| 文件 | 职责 |
|------|------|
| `scripts/eval_rag.py` | 用 LLM-as-Judge 自动评估 RAG 回答质量 |
| `eval_questions.csv` | 评估测试集（12 道知识库内 + 4 道知识库外） |
| `scripts/visualize_report.py` | 生成评估结果的 HTML 报告 |
| `eval_report.html` | 评估报告（运行 visualize_report.py 后生成） |
| `scripts/batch_upload.py` | 批量上传 `data/` 目录下所有文档 |

---

## 核心设计要点

### 分类重试策略

`rag_chain.py` 中的 `is_retryable_exception` 函数区分了"值得重试"和"不值得重试"的异常：

-  **重试**：超时、429（限流）、502/503/504（服务端临时故障）
-  **不重试**：400（参数错误）、401（密钥错误）—— 重试多少次都没用

### 向量距离换算

Chroma 的 Cosine 模式下返回的 dist 是余弦距离（`dist = 1 - cos_sim`），因此 `similarity = 1 - dist` 就是余弦相似度。collection 创建时通过 `metadata={"hnsw:space": "cosine"}` 显式声明距离度量。

### 流式输出中的元数据协议（真 SSE）

流式接口 `/ask/stream` 返回标准 `text/event-stream`，用三种事件分离"正文"与"来源"：先推 `sources`（检索/重排结果，含文件名元数据），再逐段推 `delta`（正文增量），最后以 `done`（携带 session_id）收尾。前端按事件类型分别渲染，不存在解析歧义。

### Agent 状态机的 reducer

LangGraph 中 `AgentState` 的 `messages` 字段通过 `Annotated[list, operator.add]` 声明了归约器，确保每轮返回的新消息是**追加**到历史列表而非替换。缺失此设置会导致消息链断裂、API 报错 —— 详见 [bug 修复记录](./agent_bug_fix_record.txt)。

---

## 项目演进历史

| 版本 | 提交 | 主要内容 |
|------|------|---------|
| v1.0 | `092c248` | 基础 RAG：文档上传 -> 向量检索 -> LLM 回答 |
| v2.0 | `5ea43cb` | 多策略切分、Rerank、查询改写、引用高亮、评估系统 |
| v2.1 | `b44b9c2` | 全面异步重构、修复 LangGraph reducer Bug |
| v2.2 | `03bec82` | 可视化评估报告 |
| v2.3 | `9d77be2`、`e79fa5a` | 完善工程化，添加 Dockerfile |

---


---

> **提示**：`.env` 文件包含真实 API 密钥，请勿提交到 git。`chroma_data/` 为向量数据库持久化目录，删除后重新上传文档即可重建。
