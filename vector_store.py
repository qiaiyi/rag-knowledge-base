import chromadb
from embedding_util import get_embedding

class KnowledgeBase:
    def __init__(self, collection_name="my_docs", persist_directory="./chroma_data"):
        """
        初始化知识库
        :param collection_name: 集合名称（类似数据库中的表名）
        :param persist_directory: 持久化目录，数据将保存到磁盘
        """
        # 使用 PersistentClient 让数据持久化（重启程序不丢失）
        self.client = chromadb.PersistentClient(path=persist_directory)
        # 尝试获取已有的集合，若不存在则创建新集合
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(collection_name)

    def add_documents(self, chunks, metadatas=None):
        """
        将文档块存入向量库
        :param chunks: 文本块列表，例如 ["文本1", "文本2", ...]
        :param metadatas: 元数据列表，每个元素是一个字典，例如 [{"source": "file1.txt"}, ...]
        """
        # 1. 为每个文本块生成 embedding 向量
        embeddings = [get_embedding(chunk) for chunk in chunks]
        
        # 2. 为每个文本块生成唯一 ID
        #    实际项目中可以用哈希值或文件名+序号，这里简单用递增数字
        ids = [f"doc_{i}" for i in range(len(chunks))]
        
        # 3. 如果没有提供元数据，则创建空元数据列表
        if metadatas is None:
            metadatas = [{}] * len(chunks)
        
        # 4. 添加到 Chroma 集合中
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        print(f"✅ 已添加 {len(chunks)} 个文档块到知识库")

    def search(self, query, top_k=3):
        """
        检索与查询最相关的 top_k 个文档块
        :param query: 用户问题字符串
        :param top_k: 返回的最大结果数
        :return: 文档内容列表（按相关度降序）
        """
        # 1. 将用户问题转为查询向量
        query_embedding = get_embedding(query)
        
        # 2. 在向量库中搜索最相似的 top_k 个向量
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # results 结构示例：
        # {
        #   'documents': [['文本块1', '文本块2', ...]],
        #   'distances': [[0.12, 0.45, ...]],
        #   'metadatas': [[{}, {}, ...]],
        #   'ids': [['id1', 'id2', ...]]
        # }
        # 返回第一个查询（只有一个）的 documents 列表
        return results["documents"][0] if results["documents"] else []

    def search_with_details(self, query, top_k=3):
        """返回详细信息（包括内容、距离、元数据）"""
        query_embedding = get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )
        return results

# 简单测试（直接运行此文件时执行）
if __name__ == "__main__":
    # 创建一个名为 test_kb 的知识库（数据保存在 ./chroma_data 目录）
    kb = KnowledgeBase("test_kb")
    
    # 添加几个示例文档块
    kb.add_documents([
        "北京是中国的首都，位于华北平原北部，历史悠久。",
        "Python 是一种高级编程语言，由 Guido van Rossum 创建。",
        "大语言模型（LLM）是人工智能的重要分支，基于 Transformer 架构。"
    ], metadatas=[
        {"source": "地理知识.txt"},
        {"source": "编程知识.txt"},
        {"source": "AI知识.txt"}
    ])
    
    # 检索测试
    query = "哪种编程语言比较流行？"
    print(f"\n问题：{query}")
    results = kb.search(query, top_k=2)
    print("检索结果：")
    for idx, r in enumerate(results, 1):
        print(f"  {idx}. {r}")