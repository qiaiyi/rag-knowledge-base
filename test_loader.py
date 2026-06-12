from document_loader import load_document

# 测试 TXT
content = load_document("test.txt")
print("TXT 前200字符:", content[:200])

# 测试 PDF
content = load_document("test.pdf")
print("PDF 前200字符:", content[:200])

# 测试 DOCX
content = load_document("test.docx")
print("DOCX 前200字符:", content[:200])