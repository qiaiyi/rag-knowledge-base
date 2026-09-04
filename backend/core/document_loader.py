import os
from pypdf import PdfReader
from docx import Document


def load_text(file_path):
    """加载纯文本文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(file_path):
    """
    加载 PDF 文件，提取所有页面文本
    :raises ValueError: 当 PDF 为扫描件（无文本层）或提取文本过少时
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # 修复：若文本内容少于10个字符，判定为扫描件或无效PDF，主动抛出异常
    if len(text.strip()) < 10:
        raise ValueError(f"PDF 文件 '{file_path}' 未提取到有效文本内容（可能为扫描件）")
    return text


def load_docx(file_path):
    """
    加载 Word 文件，提取段落和表格内容
    """
    doc = Document(file_path)
    text = ""

    # 1. 提取段落文本（原有逻辑）
    for para in doc.paragraphs:
        if para.text:
            text += para.text + "\n"

    # 2. 修复：提取表格文本（新增逻辑）
    for table in doc.tables:
        for row in table.rows:
            row_cells = []
            for cell in row.cells:
                if cell.text:
                    row_cells.append(cell.text.strip())
            if row_cells:
                # 用制表符或竖线连接同一行的单元格，保留表格结构
                text += " | ".join(row_cells) + "\n"

    return text


def load_document(file_path):
    """
    根据后缀自动选择加载方式
    :raises FileNotFoundError: 文件不存在
    :raises ValueError: 不支持的文件格式或内容无效
    """
    # 修复：增加文件存在性预检
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return load_text(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    else:
        raise ValueError(f"不支持的文件格式：{ext}")
