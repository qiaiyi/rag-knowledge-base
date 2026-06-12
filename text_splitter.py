import re

def split_text(text, chunk_size=500, overlap=50):
    """
    按指定长度切分文本，支持重叠。
    优先在句号、问号、感叹号、换行处切分（保持句子完整）。
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap")
    
    # 用正则把文本切成“句子级”单元（包含标点）
    sentences = re.split(r'(?<=[。！？；\n])\s*', text)
    
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        # 如果加上当前句子后超出 chunk_size，且当前块非空，则保存当前块
        if len(current_chunk) + len(sent) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # 保留 overlap 长度的尾部作为下一块的开头
            # 简单起见，我们从当前块末尾取 overlap 字符
            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk = overlap_text + sent
        else:
            current_chunk += sent
    
    # 最后一块
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# 简单测试（可直接运行）
if __name__ == "__main__":
    sample = "RAG是检索增强生成。它结合了检索和生成。今天我们要学习文本切分。"
    chunks = split_text(sample, chunk_size=20, overlap=5)
    for i, c in enumerate(chunks):
        print(f"块{i+1}: {c}")