import json
import requests
import re
from rag_chain import answer_question
from vector_store import KnowledgeBase

def evaluate(test_file="eval_test_set.json", use_rerank=True, use_query_rewrite=True):
    """运行测试集，输出通过率"""
    # 加载知识库（假设已经建立并持久化）
    kb = KnowledgeBase("demo_kb")
    
    with open(test_file, "r", encoding="utf-8") as f:
        tests = json.load(f)
    
    results = []
    for test in tests:
        question = test["question"]
        expected_keywords = test["expected_keywords"]
        should_have_citation = test.get("should_contain_citation", False)
        
        answer, sources = answer_question(
            question, kb,
            use_rerank=use_rerank,
            use_query_rewrite=use_query_rewrite
        )
        
        # 检查关键词
        keyword_match = any(kw in answer for kw in expected_keywords)
        # 检查引用格式
        has_citation = bool(re.search(r'【\d+】', answer))
        citation_ok = (has_citation == should_have_citation)
        
        passed = keyword_match and citation_ok
        results.append({
            "question": question,
            "answer": answer,
            "passed": passed,
            "keyword_match": keyword_match,
            "citation_ok": citation_ok
        })
    
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"测试通过率: {passed}/{total} = {passed/total*100:.1f}%")
    for r in results:
        status = "✅" if r["passed"] else "❌"
        print(f"{status} 问题: {r['question'][:50]}")
        if not r["passed"]:
            print(f"   关键词匹配: {r['keyword_match']}, 引用格式正确: {r['citation_ok']}")
    return results

if __name__ == "__main__":
    # 可以分别测试不同配置
    print("=== 默认配置（Rerank + 查询重写）===")
    evaluate(use_rerank=True, use_query_rewrite=True)
    
    #print("\n=== 关闭重排序 ===")
    #evaluate(use_rerank=False, use_query_rewrite=True)
    
    #print("\n=== 关闭查询重写 ===")
    #evaluate(use_rerank=True, use_query_rewrite=False)