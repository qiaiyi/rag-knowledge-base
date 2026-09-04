import pytest

from backend.core.text_splitter import split_by_fixed, split_by_sentence, split_by_paragraph, split_text


class TestSplitByFixed:
    def test_basic_chunking(self):
        text = "a" * 1200
        chunks = split_by_fixed(text, chunk_size=500, overlap=100)
        assert len(chunks) >= 3
        assert all(len(c) <= 500 for c in chunks)
        assert "".join(chunks) != ""  # 不丢内容

    def test_overlap_keeps_tail(self):
        chunks = split_by_fixed("0123456789" * 30, chunk_size=100, overlap=20)
        # 相邻块之间应共享 overlap 个字符
        for prev, nxt in zip(chunks, chunks[1:]):
            assert prev[-20:] == nxt[:20]

    def test_short_text_single_chunk(self):
        assert split_by_fixed("hello", chunk_size=500, overlap=100) == ["hello"]

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            split_by_fixed("abc", chunk_size=100, overlap=100)


class TestSplitBySentence:
    def test_keeps_sentences_intact(self):
        text = "第一句话。第二句话。第三句话！第四句话？"
        chunks = split_by_sentence(text, chunk_size=20, overlap=0)
        joined = "".join(chunks)
        for sent in ["第一句话。", "第二句话。", "第三句话！", "第四句话？"]:
            assert sent in joined

    def test_respects_chunk_size(self):
        text = "。".join(["字" * 30] * 10) + "。"
        chunks = split_by_sentence(text, chunk_size=100, overlap=0)
        assert all(len(c) <= 100 for c in chunks)


class TestSplitByParagraph:
    def test_merges_short_paragraphs(self):
        text = "段落一\n\n段落二\n\n段落三"
        chunks = split_by_paragraph(text, chunk_size=100)
        assert len(chunks) == 1
        assert "段落一" in chunks[0] and "段落三" in chunks[0]

    def test_hard_limit(self):
        text = "\n\n".join(["段" * 60] * 6)
        chunks = split_by_paragraph(text, chunk_size=100)
        assert all(len(c) <= 100 for c in chunks)
        assert len(chunks) >= 3


class TestSplitTextDispatch:
    def test_unknown_method(self):
        with pytest.raises(ValueError, match="未知切分方法"):
            split_text("abc", method="magic")

    def test_empty_text(self):
        # 空文本返回空列表或仅含空白字符串的块
        for method in ("fixed", "sentence", "paragraph"):
            chunks = split_text("", method=method)
            assert all(not c.strip() for c in chunks)

    def test_dispatch_matches_impl(self):
        text = "测试文本" * 50
        assert split_text(text, chunk_size=100, overlap=10, method="fixed") == \
            split_by_fixed(text, chunk_size=100, overlap=10)
