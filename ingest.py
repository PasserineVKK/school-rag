import os
import shutil
from config import DATA_DIR, CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, HF_CACHE_DIR
import tempfile

# Chuyển thư mục tạm sang ổ D (hoặc ổ bạn có dung lượng trống)
TEMP_DIR = r"D:\temp_hf"          # ← Thay D: thành ổ bạn muốn
os.makedirs(TEMP_DIR, exist_ok=True)

os.environ["TMP"] = TEMP_DIR
os.environ["TEMP"] = TEMP_DIR
os.environ["TMPDIR"] = TEMP_DIR
tempfile.tempdir = TEMP_DIR

print(f"📁 Thư mục tạm đã chuyển sang: {TEMP_DIR}")

print("🚀 Bắt đầu ingest dữ liệu...")

# Tạo thư mục cache nếu chưa có
os.makedirs(HF_CACHE_DIR, exist_ok=True)
print(f"📁 Cache model sẽ được lưu tại: {HF_CACHE_DIR}")

from langchain_huggingface import HuggingFaceEmbeddings
from utils import load_and_split_pdf
from langchain_chroma import Chroma

# Load embedding (chỉ tải lần đầu tiên)
print(f"🔄 Đang load embedding model: {EMBEDDING_MODEL}")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
    cache_folder=HF_CACHE_DIR
)

# Xóa DB cũ để rebuild
if os.path.exists(CHROMA_PERSIST_DIR):
    shutil.rmtree(CHROMA_PERSIST_DIR)
    print("🗑️  Đã xóa vector DB cũ.")

all_docs = []
pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

print(f"📄 Tìm thấy {len(pdf_files)} file PDF")

for pdf_file in pdf_files:
    pdf_path = os.path.join(DATA_DIR, pdf_file)
    print(f"Đang xử lý: {pdf_file}")
    try:
        chunks = load_and_split_pdf(pdf_path)
        all_docs.extend(chunks)
        print(f"   → {len(chunks)} chunks")
    except Exception as e:
        print(f"❌ Lỗi {pdf_file}: {e}")

# Tạo vector store
vectorstore = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    persist_directory=CHROMA_PERSIST_DIR,
    collection_name=COLLECTION_NAME
)

print(f"\n✅ Hoàn thành! Đã ingest {len(all_docs)} chunks từ {len(pdf_files)} file PDF.")
print(f"💾 Model cache được lưu tại: {HF_CACHE_DIR}")
print("Bạn có thể chạy: streamlit run app.py")