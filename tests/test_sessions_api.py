from fastapi.testclient import TestClient

import main

client = TestClient(main.app)
AUTH = {"X-API-Key": "test-admin-key"}


class TestAskWithSession:
    def test_multi_turn_history_passed_to_chain(self, monkeypatch):
        captured = {}

        async def fake_answer(question, kb, **kwargs):
            captured["question"] = question
            captured["history"] = kwargs.get("history")
            return "RAG 的作用是减少幻觉。", [{"content": "片段", "source": "a.txt", "score": 0.9}]

        monkeypatch.setattr(main, "answer_question", fake_answer)

        # 第一轮：不带 session_id，后端应新建会话并在响应中返回
        r1 = client.post("/ask?question=什么是RAG", headers=AUTH)
        assert r1.status_code == 200
        sid = r1.json()["session_id"]
        assert sid
        assert captured["history"] == []  # 新会话无历史

        # 第二轮：追问指代词，应携带第一轮的问答历史
        r2 = client.post(f"/ask?question=它的作用是什么&session_id={sid}", headers=AUTH)
        assert r2.status_code == 200
        history = captured["history"]
        assert {"role": "user", "content": "什么是RAG"} in history
        assert {"role": "assistant", "content": "RAG 的作用是减少幻觉。"} in history

        # 响应携带来源元数据
        assert r2.json()["sources"][0]["source"] == "a.txt"

    def test_messages_persisted_after_ask(self, monkeypatch):
        async def fake_answer(question, kb, **kwargs):
            return "回答", []

        monkeypatch.setattr(main, "answer_question", fake_answer)

        sid = client.post("/ask?question=问题A", headers=AUTH).json()["session_id"]
        msgs = main.store.get_messages(sid)
        assert [(m["role"], m["content"]) for m in msgs] == [("user", "问题A"), ("assistant", "回答")]

    def test_ask_missing_session_404(self):
        r = client.post("/ask?question=hi&session_id=no-such", headers=AUTH)
        assert r.status_code == 404


class TestSessionEndpoints:
    def _make_session(self, monkeypatch, question="你好"):
        async def fake_answer(q, kb, **kwargs):
            return "回答", []
        monkeypatch.setattr(main, "answer_question", fake_answer)
        return client.post(f"/ask?question={question}", headers=AUTH).json()["session_id"]

    def test_list_sessions(self, monkeypatch):
        sid = self._make_session(monkeypatch)
        r = client.get("/sessions", headers=AUTH)
        assert r.status_code == 200
        sessions = r.json()["sessions"]
        assert any(s["id"] == sid and s["message_count"] == 2 for s in sessions)

    def test_get_messages(self, monkeypatch):
        sid = self._make_session(monkeypatch, question="你好")
        r = client.get(f"/sessions/{sid}/messages", headers=AUTH)
        assert r.status_code == 200
        msgs = r.json()["messages"]
        assert msgs[0]["role"] == "user" and msgs[0]["content"] == "你好"
        assert msgs[1]["role"] == "assistant"

    def test_get_messages_missing_404(self):
        assert client.get("/sessions/no-such/messages", headers=AUTH).status_code == 404

    def test_delete_session(self, monkeypatch):
        sid = self._make_session(monkeypatch)
        assert client.delete(f"/sessions/{sid}", headers=AUTH).status_code == 200
        assert client.delete(f"/sessions/{sid}", headers=AUTH).status_code == 404
        assert client.get(f"/sessions/{sid}/messages", headers=AUTH).status_code == 404
