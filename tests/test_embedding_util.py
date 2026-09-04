import asyncio

import httpx
import pytest

from backend.core import embedding_util
from backend.core.embedding_util import get_embedding, EmbeddingError


def _patch_transport(monkeypatch, handler):
    """把 embedding_util 内部创建的 httpx.AsyncClient 换成 MockTransport 版本"""
    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(embedding_util.httpx, "AsyncClient", factory)


class TestGetEmbedding:
    def test_success(self, monkeypatch):
        _patch_transport(
            monkeypatch,
            lambda request: httpx.Response(
                200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]}
            ),
        )
        vector = asyncio.run(get_embedding("你好世界"))
        assert vector == [0.1, 0.2, 0.3]

    def test_empty_text_raises_value_error(self):
        with pytest.raises(ValueError):
            asyncio.run(get_embedding("   "))

    def test_http_error_becomes_embedding_error(self, monkeypatch):
        _patch_transport(monkeypatch, lambda request: httpx.Response(401))
        with pytest.raises(EmbeddingError, match="401"):
            asyncio.run(get_embedding("测试"))

    def test_missing_embedding_field(self, monkeypatch):
        _patch_transport(monkeypatch, lambda request: httpx.Response(200, json={"data": []}))
        with pytest.raises(EmbeddingError, match="数据结构异常"):
            asyncio.run(get_embedding("测试"))
