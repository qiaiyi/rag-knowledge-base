import requests
import os
from dotenv import load_dotenv
from vector_store import KnowledgeBase
from document_loader import load_document
from text_splitter import split_text
from reranker import rerank

load_dotenv()

# DeepSeek 聊天配置
LLM_URL = f"{os.getenv('BASE_URL')}/chat/completions"
LLM_KEY = os.getenv("API_KEY")

# 改进的 Prompt 模板（强制引用 + 防幻觉）
RAG_PROMPT_TEMPLATE = """你是一个严谨的知识助手。请根据以下【参考资料】回答问题。回答要求：
1. 如果参考资料中包含答案，请基于资料回答，并在每个使用到的事实后面用【编号】标注来源，例如【1】。
2. 如果参考资料中完全没有相关信息，请明确回答“资料中未提及”，不要编造任何内容。
3. 回答应简洁、准确，不要输出多余的解释。

【参考资料】
{context}

【问题】
{question}

【回答】"""

def rewrite_query(original_query, model="deepseek-v3-2-251201"):
    """
    利用 LLM 改写查询，提升检索效果
    :param original_query: 原始用户问题
    :param model: 使用的 LLM 模型名称
    :return: 改写后的查询字符串
    """
    prompt = f"""请将以下用户问题改写为更适合文档检索的形式。要求：
1. 如果问题已经清晰完整，直接返回原句。
2. 如果问题过于简短或口语化，请扩展为完整、清晰的查询，去除指代和模糊词。
3. 输出只包含改写后的查询，不要添加任何解释或额外文字。

原始问题：{original_query}
改写："""
    headers = {
        "Authorization": f"Bearer {LLM_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    try:
        resp = requests.post(LLM_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        rewritten = resp.json()["choices"][0]["message"]["content"].strip()
        return rewritten
    except Exception as e:
        print(f"查询重写失败: {e}")
        return original_query  # 降级：返回原问题

def ask_with_context(question, context_chunks, model="deepseek-v3-2-251201"):
    """根据检索到的上下文生成答案（带引用编号）"""
    # 给每个片段加上编号标签
    numbered_chunks = []
    for i, chunk in enumerate(context_chunks, 1):
        numbered_chunks.append(f"[{i}] {chunk}")
    context_text = "\n\n".join(numbered_chunks)
    
    # 使用改进的 Prompt 模板
    prompt = RAG_PROMPT_TEMPLATE.format(context=context_text, question=question)
    
    headers = {
        "Authorization": f"Bearer {LLM_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    try:
        resp = requests.post(LLM_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return "生成答案时遇到错误。"

def answer_question(question, kb, top_k=10, score_threshold=0.3, use_rerank=True, rerank_top_n=3, use_query_rewrite=True):
    """
    完整 RAG 流程：可选查询重写 → 检索 → 可选重排序 → 生成答案
    :param question: 用户原始问题
    :param kb: KnowledgeBase 实例
    :param top_k: 向量检索返回的候选片段数（建议 10~20）
    :param score_threshold: 相似度阈值（0~1），低于此值的片段被过滤
    :param use_rerank: 是否启用重排序
    :param rerank_top_n: 重排序后保留的片段数量
    :param use_query_rewrite: 是否启用查询重写
    :return: (answer, sources) 答案和来源片段列表
    """
    # 0. 查询重写（可选）
    if use_query_rewrite:
        rewritten_question = rewrite_query(question)
        print(f"原始问题：{question}")
        print(f"改写后：{rewritten_question}")
    else:
        rewritten_question = question
    
    # 1. 向量检索（带分数过滤）
    results = kb.search_with_scores(rewritten_question, top_k=top_k, score_threshold=score_threshold)
    if not results:
        return "知识库中没有找到相关内容。", []
    
    chunks = [doc for doc, _ in results]
    print(f"向量检索得到 {len(chunks)} 个片段，最高相似度: {results[0][1]:.3f}")
    
    # 2. 重排序（可选）
    if use_rerank and len(chunks) > 1:
        chunks = rerank(rewritten_question, chunks, top_n=rerank_top_n)
        print(f"重排序后保留 {len(chunks)} 个片段")
    
    # 3. 生成答案（使用原始问题，答案更自然）
    answer = ask_with_context(question, chunks)
    return answer, chunks

if __name__ == "__main__":
    # 1. 加载文档
    try:
        docs = load_document("test.txt")
    except FileNotFoundError:
        print("请先创建 test.txt 文件")
        exit(1)
    
    # 2. 切分文档
    chunks = split_text(docs)   # 默认 chunk_size=500, overlap=100
    print(f"文档已切分为 {len(chunks)} 个块")
    
    # 3. 建立向量库（如果已经持久化，会尝试添加；若 ID 重复，提示用户删除 chroma_data）
    kb = KnowledgeBase("demo_kb")
    try:
        metadatas = [{"source": "test.txt"}] * len(chunks)
        kb.add_documents(chunks, metadatas=metadatas)
    except Exception as e:
        print(f"添加文档时出错（可能是ID重复）：{e}")
        print("提示：你可以删除 chroma_data 文件夹后重新运行。")
    
    # 4. 提问（可以修改问题，测试查询重写、重排序等）
    query = "那个写 Python 的人是谁？"
    answer, sources = answer_question(
        query, kb,
        top_k=10,
        score_threshold=0.3,
        use_rerank=True,
        rerank_top_n=3,
        use_query_rewrite=True
    )
    print("\n检索到的来源：")
    for i, src in enumerate(sources, 1):
        print(f"{i}. {src[:100]}...")
    print(f"\n最终答案：{answer}")