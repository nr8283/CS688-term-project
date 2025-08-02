import streamlit as st
from faq_assistant import search_index, ask_openai, ask_ollama, normalize_query, load_faiss, TOP_K, MEMORY_TURNS

# Load FAISS and embedding model
@st.cache_resource
def load_resources():
    return load_faiss()

model, index, metadata, chunks = load_resources()

# ====== Sidebar Settings ======
st.sidebar.title("⚙️ Settings")

# Model selection
model_choice = st.sidebar.radio(
    "Select model",
    ("OpenAI", "Ollama"),
    index=0  # OpenAI as default
)

# Top K Chunks
top_k = st.sidebar.slider("🔍 Top K Chunks", 3, 10, TOP_K)

# Conversation Memory Turns
memory_turns = st.sidebar.slider("🧠 Conversation Memory", 0, 5, MEMORY_TURNS)

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []

# ====== Main UI ======
st.title("🎓 BU Graduate Programs FAQ Assistant")
st.write("Ask me anything about BU MET graduate programs.")

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"<div style='text-align:right; color:white; background-color:#0066cc; padding:8px; border-radius:10px; margin:5px 0;'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color:black; background-color:#f1f1f1; padding:8px; border-radius:10px; margin:5px 0;'>{message['content']}</div>", unsafe_allow_html=True)
        if "sources" in message and message["sources"]:
            with st.expander("📄 Sources"):
                for src in message["sources"]:
                    st.write(f"- {src}")

# ====== Input form to avoid infinite loop ======
with st.form(key="chat_form", clear_on_submit=True):
    query = st.text_input("Enter your question:", key="user_input")
    submitted = st.form_submit_button("Send")

if submitted and query:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Normalize query for better matching
    norm_query = normalize_query(query)

    # Build conversation history string
    history_context = ""
    for msg in st.session_state.messages[-memory_turns * 2:]:
        if msg["role"] == "user":
            history_context += f"User: {msg['content']}\n"
        else:
            history_context += f"Assistant: {msg['content']}\n"

    # Retrieve relevant context
    context_chunks = search_index(norm_query, model, index, metadata, chunks, top_k=top_k)
    context_text = "\n".join([c["text"] for c in context_chunks])

    # Combine history and new context
    full_context = f"Conversation so far:\n{history_context}\nNew context:\n{context_text}"

    # Choose model
    if model_choice == "Ollama":
        answer = ask_ollama(full_context, query)
    else:
        answer = ask_openai(full_context, query)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sorted(set(c["source"] for c in context_chunks))
    })

    # Rerun to update UI
    st.rerun()
