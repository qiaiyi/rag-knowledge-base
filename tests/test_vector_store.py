import asyncio

from backend.core import vector_store
from backend.core.vector_store import KnowledgeBase, _rrf_fuse


def make_kb(name: str) -> KnowledgeBase:
    """conftest 已把 PersistentClient 替换为内存版，不会落盘"""
    return KnowledgeBase(collection_name=name, persist_directory="./test_unused")


def spy_collection_get(kb, monkeypatch):
    """统计 collection.get 调用次数（验证 N+1 已消除）"""
    counter = {"count": 0}
    real_get = kb.collection.get

    def spy(*args, **kwargs):
        counter["count"] += 1
        return real_get(*args, **kwargs)

    monkeypatch.setattr(kb.collection, "get", spy)
    return counter


class TestBatchEmbedding:
    def test_add_documents_splits_into_batches(self, monkeypatch):
        kb = make_kb("batch_test")
        calls = []

        async def fake_get_embeddings(texts):
            calls.append(list(texts))
            return [[0.1, 0.2] for _ in texts]

        monkeypatch.setattr(vector_store, "get_embeddings", fake_get_embeddings)

        chunks = [f"第 {i} 个片段内容" for i in range(70)]  # 70 = 64 + 6
        asyncio.run(kb.add_documents(chunks))

        assert len(calls) == 2           # 分 2 次批量请求
        assert len(calls[0]) == 64
        assert len(calls[1]) == 6
        assert kb.collection.count() == 70


class TestDedupShortCircuit:
    def test_duplicate_chunks_skip_embedding(self, monkeypatch):
        kb = make_kb("dedup_test")
        embedding_calls = []

        async def fake_get_embeddings(texts):
            embedding_calls.append(list(texts))
            return [[0.1, 0.2] for _ in texts]

        monkeypatch.setattr(vector_store, "get_embeddings", fake_get_embeddings)
        get_counter = spy_collection_get(kb, monkeypatch)

        # 首次添加：2 个片段向量化一次，且只查一次 collection.get（无 N+1）
        chunks = ["片段A 的内容", "片段B 的内容"]
        asyncio.run(kb.add_documents(chunks))
        assert len(embedding_calls) == 1
        assert get_counter["count"] == 1

        # 重复添加相同内容 + 1 个新片段：重复内容短路，只为新片段调 embedding
        asyncio.run(kb.add_documents(chunks + ["片段C 的内容"]))
        assert embedding_calls[-1] == ["片段C 的内容"]
        assert kb.collection.count() == 3
        # 第二次调用同样只查一次 collection.get
        assert get_counter["count"] == 2

    def test_all_duplicates_skip_embedding_entirely(self, monkeypatch):
        kb = make_kb("dedup_all")
        calls = []

        async def fake_get_embeddings(texts):
            calls.append(list(texts))
            return [[0.1, 0.2] for _ in texts]

        monkeypatch.setattr(vector_store, "get_embeddings", fake_get_embeddings)

        chunks = ["完全相同的内容"]
        asyncio.run(kb.add_documents(chunks))
        assert kb.collection.count() == 1

        # 全部重复：不再调 embedding，也不会重复入库
        before = len(calls)
        asyncio.run(kb.add_documents(chunks))
        assert len(calls) == before
        assert kb.collection.count() == 1


class TestRRFFusion:
    def test_merges_rankings_and_dedupes(self):
        ranked_a = ["甲", "乙", "丙"]
        ranked_b = ["丁", "甲"]
        fused = _rrf_fuse([ranked_a, ranked_b])
        # 两路去重后并集全部出现，无重复
        assert len(fused) == len(set(fused))
        assert set(fused) == {"甲", "乙", "丙", "丁"}
        # 两路都命中的"甲"融合后排最前
        assert fused[0] == "甲"


class TestBM25KeywordRecall:
    def test_captures_literal_term_that_vector_may_miss(self, monkeypatch):
        """精确术语/编号是关键词召回的强项，验证 BM25 能把这类片段捞回来"""
        kb = make_kb("bm25_test")

        async def fake_embeddings(texts):
            return [[0.1, 0.2] for _ in texts]

        monkeypatch.setattr(vector_store, "get_embeddings", fake_embeddings)
        asyncio.run(kb.add_documents([
            "关于员工报销流程的说明文档",
            "服务器 IP 地址为 192.168.1.100，请勿泄露",
            "产品定价与优惠策略",
        ]))

        hits = kb._bm25_search("192.168.1.100", top_k=3)
        contents = [h["content"] for h in hits]
        assert contents[0] == "服务器 IP 地址为 192.168.1.100，请勿泄露"
