import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # ---------- API 密钥与地址 ----------
    BASE_URL: str = os.getenv("BASE_URL", "")
    API_KEY: str = os.getenv("API_KEY", "")
    SF_BASE_URL: str = os.getenv("SF_BASE_URL", "")
    SF_API_KEY: str = os.getenv("SF_API_KEY", "")

    # ---------- 模型名称 ----------
    LLM_MODEL: str = "deepseek-v3-2-251201"
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"

    # ---------- 文本切分 ----------
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50          # 仅用于 fixed 方法

    # ---------- 检索与重排序 ----------
    TOP_K: int = 10                 # 向量检索初选数
    SCORE_THRESHOLD: float = 0.3     # 相似度过滤阈值
    RERANK_TOP_N: int = 3            # 重排序后保留数
    USE_RERANK: bool = True
    USE_QUERY_REWRITE: bool = True

    # ---------- 超时与温度 ----------
    REQUEST_TIMEOUT: int = 30        # Embedding / Rerank 超时
    LLM_TIMEOUT: int = 60            # LLM 生成超时
    TEMPERATURE: float = 0.3

    # ---------- Prompt 模板 ----------
    RAG_PROMPT_TEMPLATE: str = """你是一个严谨的知识助手。请根据以下【参考资料】回答问题。回答要求：
1. 如果参考资料中包含答案，请基于资料回答，并在每个使用到的事实后面用【编号】标注来源，例如【1】。
2. 如果参考资料中完全没有相关信息，请明确回答"资料中未提及"，不要编造任何内容。
3. 回答应简洁、准确，不要输出多余的解释。

【参考资料】
{context}

【问题】
{question}

【回答】"""


CONFIG = Config()