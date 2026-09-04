import json

import pytest
from fastapi.testclient import TestClient

from backend import main

client = TestClient(main.app)
AUTH = {"X-API-Key": "test-admin-key"}


def parse_sse_body(text: str):
    """把 TestClient 收到的响应体解析成 [(event, data), ...]"""
    events = []
    for block in text.strip().split("\n\n"):
        event, data_str = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: "):].strip()
            elif line.startswith("data: "):
                data_str = line[len("data: "):]
        if event and data_str is not None:
            events.append((event, json.loads(data_str)))
    return events


@pytest.fixture()
def fake_rag(monkeypatch):
    """替换检索与流式生成，隔离外部 LLM API"""
    hits = [
        {"content": "RAG 是检索增强生成。", "source": "ai.txt", "score": 0.92},
        {"content": "RAG 可减少幻觉。", "source": "ai.txt", "score": 0.85},
    ]

    async def fake_retrieve(question, kb, **kwargs):
        return hits

    async def fake_stream(question, chunks, model=None):
        yield "RAG 是"
        yield "检索增强生成。"

    monkeypatch.setattr(main, "retrieve_chunks_detailed", fake_retrieve)
    monkeypatch.setattr(main, "stream_answer_with_context", fake_stream)
    return hits


class TestSSEFormat:
    def test_content_type_is_event_stream(self, fake_rag):
        r = client.post("/ask/stream?question=什么是RAG", headers=AUTH)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-cache"

    def test_event_sequence_sources_delta_done(self, fake_rag):
        r = client.post("/ask/stream?question=什么是RAG", headers=AUTH)
        events = parse_sse_body(r.text)
        names = [e for e, _ in events]

        assert names[0] == "sources"       # 来源先于正文
        assert "delta" in names
        assert names[-1] == "done"         # done 收尾

        sources_payload = events[0][1]
        assert sources_payload[0]["content"] == "RAG 是检索增强生成。"
        assert sources_payload[0]["source"] == "ai.txt"  # 含文件名元数据

        deltas = [d["text"] for e, d in events if e == "delta"]
        assert deltas == ["RAG 是", "检索增强生成。"]

    def test_no_magic_string_in_stream(self, fake_rag):
        r = client.post("/ask/stream?question=什么是RAG", headers=AUTH)
        assert "__SOURCES__" not in r.text

    def test_empty_kb_stream(self, monkeypatch):
        async def fake_retrieve(question, kb, **kwargs):
            return []

        async def fake_stream(question, chunks, model=None):
            yield "不应被调用"
            yield ""

        monkeypatch.setattr(main, "retrieve_chunks_detailed", fake_retrieve)
        monkeypatch.setattr(main, "stream_answer_with_context", fake_stream)

        r = client.post("/ask/stream?question=冷门问题", headers=AUTH)
        events = parse_sse_body(r.text)
        names = [e for e, _ in events]
        assert names == ["sources", "delta", "done"]
        deltas = [d["text"] for e, d in events if e == "delta"]
        assert deltas == ["知识库中暂时没有相关内容。"]


class TestStreamPersistsSession:
    def test_messages_saved_after_stream(self, fake_rag):
        r = client.post("/ask/stream?question=什么是RAG", headers=AUTH)
        events = parse_sse_body(r.text)
        done = [d for e, d in events if e == "done"][0]
        session_id = done["session_id"]
        assert session_id

        msgs = main.store.get_messages(session_id)
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[0]["content"] == "什么是RAG"
        assert msgs[1]["content"] == "RAG 是检索增强生成。"
        assert msgs[1]["sources"][0]["source"] == "ai.txt"

    def test_existing_session_accumulates(self, fake_rag):
        r1 = client.post("/ask/stream?question=问题一", headers=AUTH)
        sid = parse_sse_body(r1.text)[-1][1]["session_id"]
        r2 = client.post(f"/ask/stream?question=问题二&session_id={sid}", headers=AUTH)
        assert r2.status_code == 200

        msgs = main.store.get_messages(sid)
        assert [m["content"] for m in msgs] == ["问题一", "RAG 是检索增强生成。", "问题二", "RAG 是检索增强生成。"]

    def test_stream_missing_session_404(self, fake_rag):
        r = client.post("/ask/stream?question=hi&session_id=no-such", headers=AUTH)
        assert r.status_code == 404
