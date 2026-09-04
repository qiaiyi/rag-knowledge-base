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
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # ---------- 模型名称 ----------
    LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-v3-2-251201")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    # ---------- 文本切分 ----------
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    CHUNK_METHOD: str = os.getenv("CHUNK_METHOD", "fixed")

    # ---------- 检索与重排序 ----------
    TOP_K: int = int(os.getenv("TOP_K", "10"))
    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.3"))
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "3"))
    USE_RERANK: bool = os.getenv("USE_RERANK", "True").lower() == "true"
    USE_QUERY_REWRITE: bool = os.getenv("USE_QUERY_REWRITE", "True").lower() == "true"

    # ---------- 超时与温度 ----------
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))   # Embedding / Rerank 超时
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))          # LLM 生成超时
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))

    # ========== 新增：Agent 专用配置 ==========
    # LLM 输出最大 token 数（控制生成长度）
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))

    # Agent 最大迭代步数（防止死循环）
    AGENT_RECURSION_LIMIT: int = int(os.getenv("AGENT_RECURSION_LIMIT", "15"))

    # Agent 系统提示词（用于控制行为风格）
    SYSTEM_PROMPT: str = os.getenv(
        "SYSTEM_PROMPT",
        "你是一个智能助手，可以调用工具回答问题。"
        "如果知识库和搜索结果都不足，请明确告知用户信息缺失。"
        "使用工具时，请严格遵循 JSON 格式，不要虚构参数。"
    )

    # Tavily 搜索参数
    TAVILY_SEARCH_DEPTH: str = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")  # "basic" 或 "advanced"
    TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
    TAVILY_TOPIC: str = os.getenv("TAVILY_TOPIC", "general")  # "general" | "news" | "finance"
    TAVILY_TIME_RANGE: str = os.getenv("TAVILY_TIME_RANGE", "")  # "day" | "week" | "month" | "year" | ""(不限)

    # API 重试次数（针对网络抖动）
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))

    # 多轮对话：检索改写时携带的最近消息条数
    HISTORY_MAX_MESSAGES: int = int(os.getenv("HISTORY_MAX_MESSAGES", "6"))

    # ---------- Prompt 模板（保留原有 RAG 模板，新增 Agent 模板可选） ----------
    RAG_PROMPT_TEMPLATE: str = os.getenv(
        "RAG_PROMPT_TEMPLATE",
        """你是一个严谨的知识助手。请根据以下【参考资料】回答问题。回答要求：
1. 如果参考资料中包含答案，请基于资料回答，并在每个使用到的事实后面用【编号】标注来源，例如【1】。
2. 如果参考资料中完全没有相关信息，请明确回答"资料中未提及"，不要编造任何内容。
3. 回答应简洁、准确，不要输出多余的解释。

【参考资料】
{context}

【问题】
{question}

【回答】"""
    )


CONFIG = Config()
