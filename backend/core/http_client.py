import httpx

from backend.config import CONFIG

# 全局共享的 httpx.AsyncClient（连接池复用，避免每次请求都重建 TCP/TLS 连接）
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    """惰性创建全局共享客户端。各调用点通过请求级 timeout 参数覆盖默认超时。"""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=CONFIG.REQUEST_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
    return _client


async def close_client():
    """应用关闭时释放连接池"""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
