import os
import tempfile
import uuid

# 必须在导入任何业务模块之前设置：
# config.py 在类定义时读取环境变量，tools.py 导入时实例化 Tavily 客户端，
# main.py 导入时打开 SQLite 会话库
TEST_ENV = {
    "ADMIN_API_KEY": "test-admin-key",
    "API_KEY": "test-llm-key",
    "BASE_URL": "https://example.invalid/v1",
    "SF_API_KEY": "test-sf-key",
    "SF_BASE_URL": "https://example.invalid/v1",
    "TAVILY_API_KEY": "tvly-test-key",
    # 会话库放到系统临时目录，避免在仓库里留下 .db/.wal/.shm 文件
    "CHAT_DB_PATH": os.path.join(tempfile.gettempdir(), f"test_chat_{uuid.uuid4().hex}.db"),
}
for _k, _v in TEST_ENV.items():
    os.environ.setdefault(_k, _v)

import chromadb


# 测试不落盘：把 PersistentClient 换成内存版，
# main.py 导入时创建的 KnowledgeBase 不会在磁盘上生成 chroma_data
chromadb.PersistentClient = lambda **kwargs: chromadb.EphemeralClient()
