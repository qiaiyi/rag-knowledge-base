import asyncio
import chromadb
import hashlib
import logging
from rank_bm25 import BM25Okapi

from backend.config import CONFIG
from backend.core.embedding_util import get_embedding, get_embeddings, EMBEDDING_BATCH_SIZE, EmbeddingError

logger = logging.getLogger(__name__)


def _bigram_tokens(text):
    """把文本切成 bigram token 供 BM25 使用。
    中文本身无空格分词，bigram（每相邻两字符）对中英文都通用，够用且无需额外分词依赖。"""
    text = text.lower()
    chars = [c for c in text if not c.isspace()]
    return ["".join(chars[i:i + 2]) for i in range(len(chars) - 1)]


def _rrf_fuse(ranked_lists, k=60):
    """
    Reciprocal Rank Fusion：把多路排名列表融合成单一排名（保序）。
    :param ranked_lists: 每路都是 [content, ...]，按相关度降序
    :return: 融合后的 content 列表（按融合分降序，去重）
    """
    scores = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0) + 1.0 / (k + rank + 1)
    return [item for item, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


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
        # 混合检索用：语料快照与惰性构建的 BM25 索引（add/delete 后置 None 触发重建）
        self._corpus = None  # [{content, source}]
        self._bm25 = None

    async def add_documents(self, chunks, metadatas=None):
        """
        异步将文档块存入向量库（自动去重 + 批量向量化）
        :param chunks: 文本块列表
        :param metadatas: 元数据列表
        """
        # 1. 去重：一次性查出所有候选 id，避免逐条 get 的 N+1 问题
        ids = [hashlib.md5(chunk.encode('utf-8')).hexdigest() for chunk in chunks]
        existing = await asyncio.to_thread(self.collection.get, ids=ids)
        existing_ids = set(existing["ids"])

        new_ids, new_chunks, new_metadatas = [], [], []
        for i, (cid, chunk) in enumerate(zip(ids, chunks)):
            if cid not in existing_ids:
                new_ids.append(cid)
                new_chunks.append(chunk)
                # chromadb 拒绝空 dict 元数据，无元数据时用 None 占位
                new_metadatas.append(metadatas[i] if metadatas else None)

        if not new_ids:
            logger.info(f"所有文档块均已存在，无需添加（共 {len(chunks)} 块）")
            return

        # 2. 批量生成向量（按 EMBEDDING_BATCH_SIZE 分批请求）
        try:
            embeddings = []
            for start in range(0, len(new_chunks), EMBEDDING_BATCH_SIZE):
                batch = new_chunks[start:start + EMBEDDING_BATCH_SIZE]
                embeddings.extend(await get_embeddings(batch))
        except EmbeddingError as e:
            raise RuntimeError(f"生成向量失败: {e}") from e

        # 3. 包装同步 add 操作
        await asyncio.to_thread(
            self.collection.add,
            embeddings=embeddings,
            documents=new_chunks,
            ids=new_ids,
            metadatas=new_metadatas
        )
        self._invalidate_bm25()  # 语料变化，下次检索时重建 BM25
        logger.info(f"成功添加 {len(new_ids)} 个新文档块（跳过 {len(chunks) - len(new_ids)} 个重复块）")

    # ---------- 混合检索：语料快照 + BM25 关键词索引（惰性构建） ----------
    def _invalidate_bm25(self):
        """语料变化后置空缓存，下次检索时按最新集合重建"""
        self._corpus = None
        self._bm25 = None

    def _load_corpus(self):
        """从集合一次性取出所有片段做语料快照（含来源元数据），供 BM25 与关键词召回使用"""
        if self._corpus is not None:
            return
        results = self.collection.get(include=["documents", "metadatas"])
        documents = results["documents"] or []
        metadatas = results["metadatas"] or []
        self._corpus = [
            {"content": d, "source": (m or {}).get("source")}
            for d, m in zip(documents, metadatas)
        ]

    def _get_bm25(self):
        """惰性构建 BM25 索引（语料有变化才重建）"""
        self._load_corpus()
        if self._bm25 is None and self._corpus:
            tokenized = [_bigram_tokens(item["content"]) for item in self._corpus]
            self._bm25 = BM25Okapi(tokenized)
        return self._bm25

    def _bm25_search(self, query, top_k=10):
        """关键词召回：用 BM25 对语料打分，返回 top_k 命中（含 content/source）"""
        bm25 = self._get_bm25()
        if bm25 is None:
            return []
        tokens = _bigram_tokens(query)
        scores = bm25.get_scores(tokens)
        # 按 BM25 分降序取前 top_k，但排除得分为 0（无共同 bigram）的片段
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"content": self._corpus[i]["content"], "score": 0.0, "source": self._corpus[i]["source"]}
            for i in top if scores[i] > 0
        ]

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
        异步检索并返回文档内容、相似度分数和来源元数据（经过阈值过滤）。
        默认启用混合检索：向量余弦召回 + BM25 关键词召回，用 RRF 融合提高召回上界。
        :param query: 查询文本
        :param top_k: 初检返回的最大数量
        :param score_threshold: 相似度阈值（0~1），低于该值的向量命中将被过滤
        :return: 列表，每个元素为 {"content", "score", "source"}
        """
        query_embedding = await get_embedding(query)
        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[query_embedding],
            n_results=max(top_k, CONFIG.HYBRID_TOP_K),
            include=["documents", "distances", "metadatas"]
        )
        dense_hits = []
        for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
            # Chroma 余弦距离：dist = 1 - cos_sim，范围 [0, 2]
            # 因此 similarity = 1 - dist = cos_sim，范围 [-1, 1]
            similarity = 1 - dist
            if similarity >= score_threshold:
                dense_hits.append({
                    "content": doc,
                    "score": similarity,
                    "source": (meta or {}).get("source"),
                })

        # 非混合模式：直接返回向量召回结果
        if not CONFIG.USE_HYBRID:
            return dense_hits[:top_k]

        # 关键词召回（BM25）
        bm25_hits = await asyncio.to_thread(self._bm25_search, query, CONFIG.HYBRID_TOP_K)
        if not bm25_hits:
            return dense_hits[:top_k]

        # RRF 融合：两路各自按相关度排序的 content 列表
        fused = _rrf_fuse(
            [[h["content"] for h in dense_hits], [h["content"] for h in bm25_hits]]
        )
        dense_by_content = {h["content"]: h for h in dense_hits}
        bm25_by_content = {h["content"]: h for h in bm25_hits}
        hits = []
        for content in fused:
            if content in dense_by_content:
                hits.append(dense_by_content[content])  # 保留真实相似度分数
            else:
                only_bm25 = bm25_by_content[content]
                # BM25 独有命中没有向量分数，给一个"恰好过阈值"的占位分，避免被下游误判为高度相关
                only_bm25["score"] = score_threshold
                hits.append(only_bm25)
        return hits[:top_k]

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
        self._invalidate_bm25()  # 语料变化，下次检索时重建 BM25
        logger.info(f"删除文档 '{source}' 的 {len(ids)} 个片段")
        return len(ids)
