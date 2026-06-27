import os
import httpx
import asyncio
from pathlib import Path
from dotenv import load_dotenv  # 必须导入


load_dotenv()  


# ===== 配置 =====
API_BASE_URL = "http://127.0.0.1:8000"
UPLOAD_URL = f"{API_BASE_URL}/upload"


API_KEY = os.getenv("ADMIN_API_KEY")
if API_KEY is None:
    raise ValueError("❌ 请先在 .env 中配置 ADMIN_API_KEY，或在脚本中硬编码临时密码")

DATA_FOLDER = "./data"  # 把所有文档放这个文件夹里

# ===== 批量上传函数 =====
async def upload_file(file_path: str):
    """异步上传单个文件"""
    headers = {"X-API-Key": API_KEY}
    file_name = os.path.basename(file_path)
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(file_path, "rb") as f:
                files = {"file": (file_name, f, "application/octet-stream")}
                resp = await client.post(UPLOAD_URL, files=files, headers=headers)
                if resp.status_code == 200:
                    print(f"✅ 上传成功: {file_name}")
                else:
                    print(f"❌ 上传失败: {file_name} -> {resp.text}")
    except Exception as e:
        print(f"❌ 上传异常: {file_name} -> {e}")

async def main():
    # 扫描文件夹下所有支持的文件
    supported_ext = (".pdf", ".txt", ".docx")
    files_to_upload = []
    
    for ext in supported_ext:
        files_to_upload.extend(Path(DATA_FOLDER).glob(f"*{ext}"))
    
    if not files_to_upload:
        print(f"⚠️ 在 {DATA_FOLDER} 下未找到任何支持的文件")
        return
    
    print(f"📦 发现 {len(files_to_upload)} 个文件，开始批量上传...")
    
    # 并发上传（注意控制并发数，别把 ChromaDB 搞崩）
    tasks = [upload_file(str(f)) for f in files_to_upload]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())