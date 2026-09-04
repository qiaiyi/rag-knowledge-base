import pytest
from fastapi.testclient import TestClient

import main  # noqa: E402  conftest 已先设置好环境变量

client = TestClient(main.app)
AUTH = {"X-API-Key": "test-admin-key"}


class TestPublicEndpoints:
    """根路径和健康检查不应要求 API Key（供探针使用）"""

    def test_root_open(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "RAG" in r.json()["message"]

    def test_health_open(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "healthy"}


class TestAuth:
    PROTECTED = ["/upload", "/ask", "/ask/stream", "/agent/react"]

    def test_missing_key_rejected(self):
        for path in self.PROTECTED:
            r = client.post(path)
            assert r.status_code in (401, 403, 422), f"{path} 未鉴权: {r.status_code}"

    def test_wrong_key_rejected(self):
        for path in self.PROTECTED:
            r = client.post(path, headers={"X-API-Key": "wrong-key"})
            assert r.status_code == 403, f"{path} 错误密钥未被拒绝: {r.status_code}"


class TestAskValidation:
    def test_empty_question(self):
        r = client.post("/ask?question=", headers=AUTH)
        assert r.status_code == 400

    def test_empty_question_stream(self):
        r = client.post("/ask/stream?question=  ", headers=AUTH)
        assert r.status_code == 400

    def test_agent_empty_question(self):
        r = client.post("/agent/react", headers=AUTH, json={"question": "  "})
        assert r.status_code == 400


class TestUploadValidation:
    def test_unsupported_extension(self):
        r = client.post(
            "/upload",
            headers=AUTH,
            files={"file": ("evil.exe", b"malware", "application/octet-stream")},
        )
        assert r.status_code == 400
        assert "不支持的文件类型" in r.json()["detail"]
