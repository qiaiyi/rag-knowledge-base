import asyncio
import chromadb
import hashlib
import logging
from backend.core.embedding_util import get_embedding, EmbeddingError

logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self, collection_name="my_docs", persist_directory="./chroma_data"):
        """
        初始化知识库
        :param collection_name: 集合名称（类似数据库中的表名）
        :param persist_directory: 持久化目录，数据将保存到磁盘
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            # metadata 中声明 "hnsw:space": "cosine"，告诉 Chroma 使用余弦距离
            # 如果不声明，Chroma 默认用 L2（欧氏距离），后续相似度换算公式就不对
            self.collection = self.client.create_collection(
                collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    async def add_documents(self, chunks, metadatas=None):
        """
        异步将文档块存入向量库（自动去重）
        :param chunks: 文本块列表
        :param metadatas: 元数据列表
        """
        ids = []
        new_chunks = []
        new_metadatas = []
        
        for i, chunk in enumerate(chunks):
            content_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()
            # 包装同步 get 操作
            existing = await asyncio.to_thread(self.collection.get, ids=[content_hash])
            if not existing['ids']:
                ids.append(content_hash)
                new_chunks.append(chunk)
                new_metadatas.append(metadatas[i] if metadatas else {})
        
        if ids:
            try:
                # 异步逐条生成向量（列表推导式不支持 await，需显式循环）
                embeddings = []
                for chunk in new_chunks:
                    embedding = await get_embedding(chunk)
                    embeddings.append(embedding)
            except EmbeddingError as e:
                raise RuntimeError(f"生成向量失败: {e}") from e
            
            # 包装同步 add 操作
            await asyncio.to_thread(
                self.collection.add,
                embeddings=embeddings,
                documents=new_chunks,
                ids=ids,
                metadatas=new_metadatas
            )
            logger.info(f"成功添加 {len(ids)} 个新文档块（跳过 {len(chunks) - len(ids)} 个重复块）")
        else:
            logger.info("所有文档块均已存在，无需添加")

    async def search(self, query, top_k=3):
        """
        异步检索与查询最相关的 top_k 个文档块
        :param query: 用户问题字符串
        :param top_k: 返回的最大结果数
        :return: 文档内容列表（按相关度降序）
        """
        query_embedding = await get_embedding(query)
        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results["documents"][0] if results["documents"] else []

    async def search_with_details(self, query, top_k=3):
        """异步返回详细信息（包括内容、距离、元数据）"""
        query_embedding = await get_embedding(query)
        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )
        return results

    async def search_with_scores(self, query, top_k=5, score_threshold=0.5):
        """
        异步检索并返回文档内容、相似度分数和来源元数据（经过阈值过滤）
        :param query: 查询文本
        :param top_k: 初检返回的最大数量
        :param score_threshold: 相似度阈值（0~1），低于该值的片段将被过滤
        :return: 列表，每个元素为 {"content", "score", "source"}
        """
        query_embedding = await get_embedding(query)
        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )
        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        hits = []
        for doc, dist, meta in zip(documents, distances, metadatas):
            # Chroma 余弦距离：dist = 1 - cos_sim，范围 [0, 2]
            # 因此 similarity = 1 - dist = cos_sim，范围 [-1, 1]
            # （embedding 模型输出的向量通常非负，实际范围在 [0, 1]）
            similarity = 1 - dist
            if similarity >= score_threshold:
                hits.append({
                    "content": doc,
                    "score": similarity,
                    "source": (meta or {}).get("source"),
                })
        return hits

    async def list_documents(self):
        """列出知识库中所有文档（按 source 聚合，返回文件名、片段数、最近上传时间）"""
        results = await asyncio.to_thread(
            self.collection.get, include=["metadatas"]
        )
        stats = {}
        for meta in results["metadatas"]:
            meta = meta or {}
            name = meta.get("source", "未知来源")
            entry = stats.setdefault(
                name, {"source": name, "chunks": 0, "uploaded_at": None}
            )
            entry["chunks"] += 1
            uploaded_at = meta.get("uploaded_at")
            if uploaded_at and (not entry["uploaded_at"] or uploaded_at > entry["uploaded_at"]):
                entry["uploaded_at"] = uploaded_at
        return sorted(stats.values(), key=lambda d: d["source"])

    async def delete_document(self, source):
        """
        删除指定来源文档的全部片段
        :param source: 上传时的文件名（metadata 中的 source 字段）
        :return: 实际删除的片段数；文档不存在时返回 0
        """
        existing = await asyncio.to_thread(
            self.collection.get, where={"source": source}
        )
        ids = existing["ids"]
        if not ids:
            return 0
        await asyncio.to_thread(self.collection.delete, ids=ids)
        logger.info(f"删除文档 '{source}' 的 {len(ids)} 个片段")
        return len(ids)
