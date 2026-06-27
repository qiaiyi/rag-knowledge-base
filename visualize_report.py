import pandas as pd
import os

# 读取已有评估结果
csv_path = "eval_results.csv"
if not os.path.exists(csv_path):
    print(f"❌ 未找到 {csv_path}，请先运行 eval_rag.py 生成评估结果。")
    exit(1)

df = pd.read_csv(csv_path)

# ========== 统计信息 ==========
avg = df['score'].mean()
min_score = df['score'].min()
max_score = df['score'].max()
count = len(df)
low_scores = df[df['score'] < 4]

print(f"样本数: {count}")
print(f"平均分: {avg:.2f}/5")
print(f"最高分: {max_score}")
print(f"最低分: {min_score}")
print(f"低分(<4)数量: {len(low_scores)}")

# ========== 生成 HTML 报告（带颜色标记） ==========
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RAG 系统评估报告</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        .summary {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .summary span {{ font-weight: bold; color: #007bff; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top; }}
        th {{ background-color: #343a40; color: white; }}
        .score-5 {{ background-color: #d4edda; }}
        .score-4 {{ background-color: #fff3cd; }}
        .score-3 {{ background-color: #ffe5d0; }}
        .score-2 {{ background-color: #f8d7da; }}
        .score-1 {{ background-color: #f5c6cb; }}
        .footer {{ margin-top: 30px; color: #6c757d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>📊 RAG 系统评估报告</h1>
    <div class="summary">
        <p>总样本数: <span>{count}</span></p>
        <p>平均分: <span>{avg:.2f}/5</span></p>
        <p>最高分: <span>{max_score}</span> | 最低分: <span>{min_score}</span></p>
        <p>低分 (<4) : <span style="color: #dc3545;">{len(low_scores)}</span> 条</p>
    </div>
    <h2>详细结果</h2>
    <table>
        <thead>
            <tr><th>问题</th><th>期望答案</th><th>系统回答</th><th>得分</th></tr>
        </thead>
        <tbody>
"""

for _, row in df.iterrows():
    score = row['score']
    cls = f"score-{score}" if score in [1,2,3,4,5] else ""
    answer = row['answer']
    if len(answer) > 150:
        answer = answer[:150] + "..."
    html_content += f"""
        <tr class="{cls}">
            <td>{row['question']}</td>
            <td>{row['expected']}</td>
            <td>{answer}</td>
            <td><strong>{score}</strong></td>
        </tr>
    """

html_content += """
        </tbody>
    </table>
    <div class="footer">生成时间: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
</body>
</html>
"""

with open("eval_report.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print("✅ HTML 报告已保存为 eval_report.html")