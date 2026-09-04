import asyncio

import httpx

import rag_chain
from rag_chain import rewrite_query, retrieve_chunks, retrieve_chunks_detailed, answer_question


def patch_llm(monkeypatch, response_text):
    """把 rag_chain 内部创建的 httpx.AsyncClient 换成 MockTransport 版本，并捕获请求体"""
    captured = {}
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"content": response_text}}]},
        )
    )
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        captured["url"] = str(kwargs.get("base_url", "")) or True

        class CaptureClient(real_client):
            async def post(self, url, **kw):
                captured["url"] = str(url)
                captured["payload"] = kw.get("json")
                return await super().post(url, **kw)

        kwargs["transport"] = transport
        return CaptureClient(*args, **kwargs)

    monkeypatch.setattr(rag_chain.httpx, "AsyncClient", factory)
    return captured


class TestRewriteWithHistory:
    def test_prompt_contains_history(self, monkeypatch):
        captured = patch_llm(monkeypatch, "RAG 的作用")
        history = [
            {"role": "user", "content": "什么是RAG"},
            {"role": "assistant", "content": "RAG 是检索增强生成"},
        ]
        result = asyncio.run(rewrite_query("它的作用是什么", history=history))

        assert result == "RAG 的作用"
        prompt = captured["payload"]["messages"][0]["content"]
        assert "什么是RAG" in prompt            # 历史问题进入改写上下文
        assert "检索增强生成" in prompt          # 历史回答进入改写上下文
        assert "它的作用是什么" in prompt

    def test_without_history_no_history_block(self, monkeypatch):
        captured = patch_llm(monkeypatch, "原始问题")
        asyncio.run(rewrite_query("原始问题"))
        prompt = captured["payload"]["messages"][0]["content"]
        # 指令文案里会提到"最近对话历史"，这里断言的是历史内容块本身不出现
        assert "【最近对话】" not in prompt


class FakeKB:
    """伪造向量库：search_with_scores 返回固定命中（含元数据）"""

    def __init__(self, hits):
        self.hits = hits
        self.captured_query = None

    async def search_with_scores(self, query, top_k=5, score_threshold=0.5):
        self.captured_query = query
        return self.hits


class TestRetrieveDetailed:
    HITS = [
        {"content": "片段甲", "score": 0.9, "source": "a.txt"},
        {"content": "片段乙", "score": 0.8, "source": "b.txt"},
    ]

    def test_returns_dict_hits(self, monkeypatch):
        async def fake_rewrite(q, model=None, history=None):
            return q

        monkeypatch.setattr(rag_chain, "rewrite_query", fake_rewrite)
        kb = FakeKB(self.HITS)
        hits = asyncio.run(retrieve_chunks_detailed("查询", kb, use_rerank=False))
        assert hits == self.HITS

    def test_rewrite_receives_history(self, monkeypatch):
        async def fake_rewrite(q, model=None, history=None):
            return "结合历史的改写查询"

        monkeypatch.setattr(rag_chain, "rewrite_query", fake_rewrite)
        kb = FakeKB(self.HITS)
        asyncio.run(retrieve_chunks_detailed("它", kb, use_rerank=False, history=[{"role": "user", "content": "X"}]))
        assert kb.captured_query == "结合历史的改写查询"  # 用改写后的查询去检索

    def test_rerank_reorders_hits(self, monkeypatch):
        async def fake_rewrite(q, model=None, history=None):
            return q

        async def fake_rerank(query, chunks, top_n=3):
            return list(reversed(chunks))  # 乙排到甲前面

        monkeypatch.setattr(rag_chain, "rewrite_query", fake_rewrite)
        monkeypatch.setattr(rag_chain, "rerank", fake_rerank)
        kb = FakeKB(self.HITS)
        hits = asyncio.run(retrieve_chunks_detailed("查询", kb, use_rerank=True))
        assert [h["content"] for h in hits] == ["片段乙", "片段甲"]
        assert hits[0]["source"] == "b.txt"  # 元数据跟着重排结果走

    def test_legacy_wrapper_returns_strings(self, monkeypatch):
        async def fake_rewrite(q, model=None, history=None):
            return q

        monkeypatch.setattr(rag_chain, "rewrite_query", fake_rewrite)
        kb = FakeKB(self.HITS)
        chunks = asyncio.run(retrieve_chunks("查询", kb, use_rerank=False))
        assert chunks == ["片段甲", "片段乙"]


class TestAnswerQuestion:
    def test_returns_sources_with_metadata(self, monkeypatch):
        hits = [
            {"content": "片段甲", "score": 0.9, "source": "a.txt"},
            {"content": "片段乙", "score": 0.8, "source": "b.txt"},
        ]

        async def fake_retrieve_detailed(*args, **kwargs):
            return hits

        async def fake_ask(question, chunks, model=None):
            return f"答案基于 {len(chunks)} 个片段"

        monkeypatch.setattr(rag_chain, "retrieve_chunks_detailed", fake_retrieve_detailed)
        monkeypatch.setattr(rag_chain, "ask_with_context", fake_ask)

        answer, sources = asyncio.run(answer_question("问题", FakeKB([])))
        assert answer == "答案基于 2 个片段"
        assert sources[0] == {"content": "片段甲", "score": 0.9, "source": "a.txt"}
