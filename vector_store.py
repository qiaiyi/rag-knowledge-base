import asyncio
import chromadb
import hashlib
import logging
from embedding_util import get_embedding, EmbeddingError

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
            self.collection = self.client.create_collection(collection_name)

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
        异步检索并返回文档内容和相似度分数（经过阈值过滤）
        :param query: 查询文本
        :param top_k: 初检返回的最大数量
        :param score_threshold: 相似度阈值（0~1），低于该值的片段将被过滤
        :return: 列表，每个元素为 (document, similarity_score)
        """
        query_embedding = await get_embedding(query)
        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances"]
        )
        documents = results["documents"][0]
        distances = results["distances"][0]
        
        filtered = []
        for doc, dist in zip(documents, distances):
            # Chroma 余弦距离 dist = 1 - cosine_similarity，范围 [0, 2]
            # 归一化至 [0, 1] 区间：similarity = 1 - dist / 2
            similarity = 1 - dist / 2
            if similarity >= score_threshold:
                filtered.append((doc, similarity))
        return filtered