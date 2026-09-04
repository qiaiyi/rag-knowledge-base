import pytest

from backend.storage import ChatStore


@pytest.fixture()
def store(tmp_path):
    s = ChatStore(str(tmp_path / "chat.db"))
    yield s
    s.close()


class TestSessions:
    def test_create_and_list(self, store):
        sid = store.create_session(title="测试会话")
        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["id"] == sid
        assert sessions[0]["title"] == "测试会话"
        assert sessions[0]["message_count"] == 0

    def test_exists(self, store):
        sid = store.create_session()
        assert store.session_exists(sid)
        assert not store.session_exists("no-such-id")

    def test_delete_missing_returns_false(self, store):
        assert store.delete_session("no-such-id") is False

    def test_delete_cascades_messages(self, store):
        sid = store.create_session()
        store.add_message(sid, "user", "你好")
        store.add_message(sid, "assistant", "你好，有什么可以帮你？")
        assert store.delete_session(sid) is True
        assert store.get_messages(sid) is None
        assert store.list_sessions() == []


class TestMessages:
    def test_roundtrip_with_sources(self, store):
        sid = store.create_session()
        sources = [{"content": "片段A", "source": "a.txt", "score": 0.9}]
        store.add_message(sid, "user", "问题")
        store.add_message(sid, "assistant", "回答", sources=sources)

        msgs = store.get_messages(sid)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user" and msgs[0]["content"] == "问题"
        assert msgs[1]["sources"] == sources  # JSON 序列化/反序列化后保持一致

    def test_get_messages_missing_session(self, store):
        assert store.get_messages("no-such-id") is None

    def test_add_message_missing_session_raises(self, store):
        with pytest.raises(ValueError):
            store.add_message("no-such-id", "user", "内容")

    def test_history_returns_last_n_chronological(self, store):
        sid = store.create_session()
        for i in range(5):
            store.add_message(sid, "user", f"问题{i}")
            store.add_message(sid, "assistant", f"回答{i}")

        history = store.get_history(sid, max_messages=4)
        # 只要最近 4 条，且按时间正序
        assert [m["content"] for m in history] == ["问题3", "回答3", "问题4", "回答4"]

    def test_history_missing_session_empty(self, store):
        assert store.get_history("no-such-id") == []
