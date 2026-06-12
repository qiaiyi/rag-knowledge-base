import os
from pypdf import PdfReader
from docx import Document

def load_text(file_path):
    """加载纯文本文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(file_path):
    """加载 PDF 文件，提取所有页面文本"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def load_docx(file_path):
    """加载 Word 文件"""
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def load_document(file_path):
    """根据后缀自动选择加载方式"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return load_text(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    else:
        raise ValueError(f"不支持的文件格式：{ext}")

