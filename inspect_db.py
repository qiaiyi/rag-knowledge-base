import asyncio
import random
from vector_store import KnowledgeBase

async def main():
    kb = KnowledgeBase("main_kb")
    
    # 获取所有文档的 ID 和内容
    # ChromaDB 的 get() 不带条件返回全部
    all_data = await asyncio.to_thread(
        kb.collection.get,
        include=["documents", "metadatas"]
    )
    
    ids = all_data.get("ids", [])
    docs = all_data.get("documents", [])
    metas = all_data.get("metadatas", [])
    
    total = len(docs)
    print(f"知识库中共有 {total} 个片段")
    
    if total == 0:
        print("知识库为空，请先上传文档。")
        return
    
    # 随机抽取 10 条（如果总数少于 10 则全部显示）
    sample_size = min(10, total)
    indices = random.sample(range(total), sample_size)
    
    print(f"\n随机抽取 {sample_size} 条片段如下：\n")
    for idx in indices:
        print(f"【片段 {idx}】")
        print(f"内容：{docs[idx][:300]}...")  # 截断长文本
        print(f"元数据：{metas[idx] if idx < len(metas) else {}}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())