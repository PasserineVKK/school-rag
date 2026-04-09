import os
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(pdf_path):
    """Parse PDF sang markdown tốt cho table"""
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    doc = Document(
        page_content=md_text,
        metadata={"source": os.path.basename(pdf_path)}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_documents([doc])

def format_sources(docs):
    """Chỉ trả về tên file PDF"""
    sources = set(doc.metadata["source"] for doc in docs if "source" in doc.metadata)
    if not sources:
        return ""
    return "\n\n**Nguồn:** " + ", ".join(sorted(list(sources)))