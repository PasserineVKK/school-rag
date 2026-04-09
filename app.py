import streamlit as st
import os
import warnings
import logging

# Chặn log rác
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Accessing __path__.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

st.set_page_config(page_title="RAG Quy định Trường", page_icon="📘", layout="centered")

st.title("🤖 Trợ lý Tra cứu Quy định Nội bộ Trường")

# ========================== IMPORT ==========================
try:
    from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, GEMINI_MODEL, MEMORY_K
    from utils import format_sources
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    st.success("✅ Hệ thống đã sẵn sàng")
except Exception as e:
    st.error(f"❌ Lỗi khởi tạo: {e}")
    st.stop()

# ========================== LOAD MODELS ==========================
@st.cache_resource
def init_models():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)
    return vectorstore.as_retriever(search_kwargs={"k": 6}), llm

retriever, llm = init_models()

# ========================== CHAT HISTORY MANAGEMENT ==========================
# Khởi tạo list chứa tin nhắn nếu chưa có
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hàm để lấy n câu hội thoại gần nhất làm ngữ cảnh (thay thế WindowMemory)
def get_chat_history_string(k=MEMORY_K):
    # Mỗi lượt hội thoại gồm 2 tin nhắn (User & AI), nên lấy k*2
    recent_messages = st.session_state.messages[-(k*2):]
    history_str = ""
    for msg in recent_messages:
        prefix = "Người dùng" if msg["role"] == "user" else "Trợ lý"
        history_str += f"{prefix}: {msg['content']}\n"
    return history_str

# ========================== PROMPT ==========================
prompt_template = ChatPromptTemplate.from_template("""
Bạn là trợ lý thông minh hỗ trợ sinh viên tra cứu quy định nội bộ trường học.
Trả lời bằng tiếng Việt, rõ ràng, lịch sự.
Chỉ dùng thông tin từ tài liệu được cung cấp. Nếu không tìm thấy thì nói rõ.

Lịch sử chat gần đây:
{chat_history}

Thông tin tài liệu:
{context}

Câu hỏi mới: {question}
Trả lời:
""")

# ========================== CHAT INTERFACE ==========================
# Hiển thị lịch sử chat lên màn hình
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Hỏi về quy định trường..."):
    # 1. Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Xử lý câu trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            try:
                # Tìm tài liệu liên quan
                docs = retriever.invoke(prompt)
                context = "\n\n".join([d.page_content for d in docs])
                
                # Lấy lịch sử dạng chuỗi
                chat_history = get_chat_history_string(k=MEMORY_K)
                
                # Chạy Chain
                chain = (
                    {"context": lambda x: context, 
                     "question": RunnablePassthrough(),
                     "chat_history": lambda x: chat_history}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )
                
                response = chain.invoke(prompt)
                sources = format_sources(docs)
                full_response = response + "\n\n" + sources
                
                # 3. Hiển thị và lưu lại
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Lỗi: {e}")

# Sidebar
with st.sidebar:
    st.header("Cấu hình")
    st.info(f"Đang nhớ {MEMORY_K} lượt chat gần nhất")
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()