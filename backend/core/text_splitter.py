import re
from backend.config import CONFIG

def split_by_fixed(text, chunk_size=500, overlap=100):
    """固定长度切分（带重叠）"""
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def split_by_sentence(text, chunk_size=500, overlap=100):
    """按句子边界切分，尽量保持句子完整"""
    # 按中文标点分割成句子列表（保留标点）
    sentences = re.split(r'(?<=[。！？；\n])\s*', text)
    chunks = []
    current = ""
    for sent in sentences:
        # 如果当前块加上新句子后超出 chunk_size，则保存当前块并开始新块
        if len(current) + len(sent) > chunk_size and current:
            chunks.append(current.strip())
            # 若有重叠需求，可将当前块末尾若干字符保留到下一块（简化）
            if overlap > 0 and len(current) >= overlap:
                current = current[-overlap:] + sent
            else:
                current = sent
        else:
            current += sent
    if current:
        chunks.append(current.strip())
    return chunks


def split_by_paragraph(text, chunk_size=500):
    """
    按段落切分：将文本按空行分割为段落，然后合并短段落，
    确保每个分块长度不超过 chunk_size（硬上限）。
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current = ""

    for para in paragraphs:
        # 如果当前块已有内容，且加入新段落会超过上限，则先保存当前块
        if len(current) + len(para) > chunk_size and current:
            chunks.append(current.strip())
            current = para + "\n\n"  # 新块从此段落开始
        else:
            current += para + "\n\n"

    # 处理末尾未保存的块
    if current:
        chunks.append(current.strip())

    return chunks


def split_text(text, chunk_size=CONFIG.CHUNK_SIZE, overlap=CONFIG.CHUNK_OVERLAP, method="fixed"):
    """统一接口，方便切换"""
    if method == "fixed":
        return split_by_fixed(text, chunk_size, overlap)
    elif method == "sentence":
        return split_by_sentence(text, chunk_size, overlap)
    elif method == "paragraph":
        return split_by_paragraph(text, chunk_size=chunk_size)
    else:
        raise ValueError(f"未知切分方法: {method}")