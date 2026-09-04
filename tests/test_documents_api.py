import uuid

from fastapi.testclient import TestClient

from backend import main


client = TestClient(main.app)
AUTH = {"X-API-Key": "test-admin-key"}


def seed_kb(source: str, n_chunks: int = 2):
    """直接向内存版 Chroma 集合写入测试数据（绕过 embedding API）"""
    main.kb.collection.add(
        ids=[f"{source}-{i}-{uuid.uuid4().hex[:6]}" for i in range(n_chunks)],
        documents=[f"{source} 的第 {i} 个片段内容" for i in range(n_chunks)],
        metadatas=[
            {"source": source, "uploaded_at": "2026-09-01T10:00:00"} for _ in range(n_chunks)
        ],
        embeddings=[[0.1, 0.2] for _ in range(n_chunks)],
    )


class TestListDocuments:
    def test_empty_kb(self):
        r = client.get("/documents", headers=AUTH)
        assert r.status_code == 200
        # 其他测试可能已写入数据，这里只验证结构
        assert isinstance(r.json()["documents"], list)

    def test_lists_source_with_chunk_count(self):
        name = f"列表测试_{uuid.uuid4().hex[:6]}.txt"
        seed_kb(name, n_chunks=3)
        r = client.get("/documents", headers=AUTH)
        docs = {d["source"]: d for d in r.json()["documents"]}
        assert name in docs
        assert docs[name]["chunks"] == 3
        assert docs[name]["uploaded_at"] == "2026-09-01T10:00:00"
        # 清理，避免影响其他测试
        client.delete(f"/documents/{name}", headers=AUTH)


class TestDeleteDocument:
    def test_delete_existing(self):
        name = f"删除测试_{uuid.uuid4().hex[:6]}.txt"
        seed_kb(name, n_chunks=2)
        r = client.delete(f"/documents/{name}", headers=AUTH)
        assert r.status_code == 200
        assert "2" in r.json()["message"]

        # 删除后列表中不再出现
        docs = client.get("/documents", headers=AUTH).json()["documents"]
        assert name not in {d["source"] for d in docs}

    def test_delete_missing_404(self):
        r = client.delete("/documents/不存在文档.txt", headers=AUTH)
        assert r.status_code == 404

    def test_delete_does_not_touch_other_docs(self):
        keep, drop = f"保留_{uuid.uuid4().hex[:6]}.txt", f"删除_{uuid.uuid4().hex[:6]}.txt"
        seed_kb(keep)
        seed_kb(drop)
        client.delete(f"/documents/{drop}", headers=AUTH)
        docs = client.get("/documents", headers=AUTH).json()["documents"]
        names = {d["source"] for d in docs}
        assert keep in names and drop not in names
        client.delete(f"/documents/{keep}", headers=AUTH)


class TestUploadAddsMetadata:
    def test_upload_records_uploaded_at(self, monkeypatch):
        import io
        from datetime import datetime

        from backend.core import vector_store

        # mock 掉批量 embedding 生成，上传流程不访问外部 API
        async def fake_embeddings(texts):
            return [[0.1, 0.2] for _ in texts]

        monkeypatch.setattr(vector_store, "get_embeddings", fake_embeddings)

        files = {"file": ("元数据测试.txt", io.BytesIO("测试内容".encode("utf-8")), "text/plain")}
        r = client.post("/upload", headers=AUTH, files=files)
        assert r.status_code == 200

        docs = client.get("/documents", headers=AUTH).json()["documents"]
        doc = next(d for d in docs if d["source"] == "元数据测试.txt")
        assert doc["uploaded_at"] is not None
        assert datetime.fromisoformat(doc["uploaded_at"])  # ISO 格式合法
        client.delete("/documents/元数据测试.txt", headers=AUTH)
