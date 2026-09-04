import asyncio
import json

import httpx
import pytest

from backend.core import embedding_util
from backend.core.embedding_util import get_embedding, get_embeddings, EmbeddingError


# 每个测试创建的 mock 客户端，测试结束后统一关闭
_CLIENTS = []


@pytest.fixture(autouse=True)
def _close_clients():
    yield
    for c in _CLIENTS:
        asyncio.run(c.aclose())
    _CLIENTS.clear()


def _patch_client(monkeypatch, handler):
    """把 embedding_util 使用的共享客户端替换为 MockTransport 版本"""
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    _CLIENTS.append(client)
    monkeypatch.setattr(embedding_util, "get_client", lambda: client)
    return client


def _capture_request(captured):
    def handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=captured.get("response", {"data": []}))
    return handler


class TestGetEmbeddingSingle:
    def test_success(self, monkeypatch):
        captured = {"response": {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}}
        _patch_client(monkeypatch, _capture_request(captured))
        vector = asyncio.run(get_embedding("你好世界"))
        assert vector == [0.1, 0.2, 0.3]
        # 单条接口也走批量 payload（input 是列表）
        assert captured["payload"]["input"] == ["你好世界"]

    def test_empty_text_raises_value_error(self):
        with pytest.raises(ValueError):
            asyncio.run(get_embedding("   "))


class TestGetEmbeddingsBatch:
    def test_success_preserves_input_order(self, monkeypatch):
        # API 返回乱序（index 1 在前），应按 index 映射回输入顺序
        captured = {"response": {"data": [
            {"index": 1, "embedding": [0.4, 0.5]},
            {"index": 0, "embedding": [0.1, 0.2]},
        ]}}
        _patch_client(monkeypatch, _capture_request(captured))
        vectors = asyncio.run(get_embeddings(["第一句", "第二句"]))
        assert vectors == [[0.1, 0.2], [0.4, 0.5]]
        assert captured["payload"]["input"] == ["第一句", "第二句"]  # 一次请求携带全部文本

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            asyncio.run(get_embeddings([]))

    def test_blank_text_in_list_raises(self):
        with pytest.raises(ValueError):
            asyncio.run(get_embeddings(["正常", "  "]))

    def test_count_mismatch_raises(self, monkeypatch):
        # 输入 2 条，API 只返回 1 条向量 → 数量不匹配
        captured = {"response": {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}}
        _patch_client(monkeypatch, _capture_request(captured))
        with pytest.raises(EmbeddingError, match="数量不匹配"):
            asyncio.run(get_embeddings(["第一条", "第二条"]))

    def test_http_error_becomes_embedding_error(self, monkeypatch):
        _patch_client(monkeypatch, lambda request: httpx.Response(401))
        with pytest.raises(EmbeddingError, match="401"):
            asyncio.run(get_embeddings(["测试"]))

    def test_missing_embedding_field(self, monkeypatch):
        _patch_client(monkeypatch, lambda request: httpx.Response(200, json={"data": []}))
        with pytest.raises(EmbeddingError, match="数据结构异常"):
            asyncio.run(get_embeddings(["测试"]))
