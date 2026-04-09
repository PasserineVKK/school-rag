from langchain_core.messages import HumanMessage, AIMessage

# Giả sử bạn muốn nhớ 5 cặp hội thoại gần nhất (k=5)
# Bạn chỉ cần cắt list tin nhắn trong session_state
k = 5
history_messages = st.session_state.messages[-(k*2):] 

# Sau đó format thành chuỗi để đưa vào prompt
history_str = "\n".join([
    f"User: {m['content']}" if m['role'] == 'user' else f"AI: {m['content']}" 
    for m in history_messages
])