import asyncio
import httpx
import pandas as pd
from config import CONFIG

# 读取测试集
test_df = pd.read_csv("eval_questions.csv")
questions = test_df["question"].tolist()
expected_answers = test_df["expected"].tolist()

async def get_rag_answer(question: str, api_key: str):
    """调用你的RAG接口获取回答"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "http://127.0.0.1:8000/ask",
            params={"question": question},
            headers={"X-API-Key": api_key}
        )
        data = resp.json()
        return data.get("answer", "")

async def judge_answer(question: str, answer: str, expected: str):
    """用LLM作为评判员打分(1-5)"""
    prompt = f"""你是一个严格的RAG系统评估专家。请根据参考答案对系统回答进行打分（1-5分）。
评分标准：
5分：回答完全正确，且覆盖所有关键信息
4分：回答正确，但遗漏次要细节
3分：回答部分正确，包含一些错误
2分：回答与参考答案相关性低
1分：回答完全错误或无关

问题：{question}
参考答案：{expected}
系统回答：{answer}

只输出分数（1-5的整数），不要其他任何文字。"""

    headers = {
        "Authorization": f"Bearer {CONFIG.API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": CONFIG.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 10
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{CONFIG.BASE_URL}/chat/completions",
            json=payload,
            headers=headers
        )
        result = resp.json()
        score_text = result["choices"][0]["message"]["content"].strip()
        try:
            return int(score_text)
        except ValueError:
            # 若输出非数字，提取第一个数字
            import re
            nums = re.findall(r"\d", score_text)
            return int(nums[0]) if nums else 3

async def main():
    api_key = input("请输入ADMIN_API_KEY: ")
    results = []
    for i, (q, exp) in enumerate(zip(questions, expected_answers), 1):
        print(f"正在评估第{i}条：{q[:20]}...")
        answer = await get_rag_answer(q, api_key)
        score = await judge_answer(q, answer, exp)
        results.append({
            "question": q,
            "expected": exp,
            "answer": answer,
            "score": score
        })
        # 延迟避免API限流
        await asyncio.sleep(1)
    
    df = pd.DataFrame(results)
    df.to_csv("eval_results.csv", index=False, encoding="utf-8-sig")
    avg_score = df["score"].mean()
    print(f"\n评估完成！平均分：{avg_score:.2f}/5")
    print(f"详细结果已保存至 eval_results.csv")

if __name__ == "__main__":
    asyncio.run(main())