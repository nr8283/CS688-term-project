# # # # import streamlit as st
# # # # import faiss
# # # # import numpy as np
# # # # from sentence_transformers import SentenceTransformer
# # # # from faq_assistant import search_index, ask_openai, ask_ollama
# # # #
# # # # # Config
# # # # USE_LOCAL = False  # change to True to use Ollama locally
# # # # LOCAL_MODEL = "mistral"
# # # #
# # # # # Load FAISS and model
# # # # @st.cache_resource
# # # # def load_resources():
# # # #     model = SentenceTransformer('all-MiniLM-L6-v2')
# # # #     index = faiss.read_index("embeddings/index.faiss")
# # # #     metadata = np.load("embeddings/metadata.npy", allow_pickle=True)
# # # #     chunks = np.load("embeddings/chunks.npy", allow_pickle=True)
# # # #     return model, index, metadata, chunks
# # # #
# # # # model, index, metadata, chunks = load_resources()
# # # #
# # # # # Streamlit app
# # # # st.set_page_config(page_title="BU Grad Programs Assistant", page_icon="🎓")
# # # # st.title("🎓 BU Graduate Programs FAQ Assistant")
# # # # st.write("Ask me anything about BU MET graduate programs.")
# # # #
# # # # if "conversation_history" not in st.session_state:
# # # #     st.session_state.conversation_history = []
# # # #
# # # # # User input
# # # # query = st.text_input("Enter your question:")
# # # #
# # # # if st.button("Ask") and query:
# # # #     # Prepare conversation context
# # # #     history_context = ""
# # # #     for q, a in st.session_state.conversation_history[-3:]:
# # # #         history_context += f"User: {q}\nAssistant: {a}\n"
# # # #
# # # #     # Search embeddings
# # # #     # context_chunks = search_index(query, model, index, metadata, top_k=5)
# # # #     context_chunks = search_index(query, model, index, metadata, chunks, top_k=5)
# # # #     context_text = "\n".join([chunk['text'] for chunk in context_chunks])
# # # #
# # # #     # Combine
# # # #     full_context = f"Conversation so far:\n{history_context}\nNew Context:\n{context_text}"
# # # #
# # # #     # Get answer
# # # #     answer = ask_ollama(full_context, query, model_name=LOCAL_MODEL) if USE_LOCAL else ask_openai(full_context, query)
# # # #
# # # #     # Save
# # # #     st.session_state.conversation_history.append((query, answer))
# # # #
# # # # # Display conversation
# # # # for q, a in reversed(st.session_state.conversation_history):
# # # #     st.markdown(f"**You:** {q}")
# # # #     st.markdown(f"**Assistant:** {a}")
# # #
# # # # src/faq_ui.py
# # # import streamlit as st
# # # from faq_assistant import (
# # #     normalize_query,
# # #     search_index,
# # #     filter_context_by_query,
# # #     ask_openai,
# # #     ask_ollama,
# # #     load_faiss
# # # )
# # #
# # # # ===== LOAD RESOURCES =====
# # # @st.cache_resource
# # # def load_resources():
# # #     return load_faiss()
# # #
# # # model, index, metadata, chunks = load_resources()
# # #
# # # # ===== STREAMLIT PAGE CONFIG =====
# # # st.set_page_config(page_title="BU Graduate Programs FAQ", layout="wide")
# # #
# # # st.title("🎓 BU Graduate Programs FAQ Assistant")
# # # st.write("Ask me anything about **BU MET** graduate programs.")
# # #
# # # # ===== SIDEBAR CONFIG =====
# # # st.sidebar.header("⚙️ Settings")
# # # model_choice = st.sidebar.radio("Select Model", ["OpenAI (gpt-4o-mini)", "Ollama (Mistral)"])
# # # top_k = st.sidebar.slider("Top K Chunks", 3, 10, 5)
# # # memory_turns = st.sidebar.slider("Conversation Memory (turns)", 0, 5, 3)
# # #
# # # # ===== SESSION STATE =====
# # # if "conversation_history" not in st.session_state:
# # #     st.session_state.conversation_history = []
# # #
# # # # ===== USER INPUT =====
# # # user_input = st.text_input("Enter your question:")
# # #
# # # if user_input:
# # #     # Normalize query for FAISS
# # #     norm_query = normalize_query(user_input)
# # #
# # #     # Build conversation memory context
# # #     history_context = ""
# # #     if st.session_state.conversation_history and memory_turns > 0:
# # #         for q, a in st.session_state.conversation_history[-memory_turns:]:
# # #             history_context += f"User: {q}\nAssistant: {a}\n"
# # #
# # #     # Search FAISS index
# # #     context_chunks = search_index(norm_query, model, index, metadata, chunks, top_k=top_k)
# # #
# # #     # Filter to keep only relevant chunks
# # #     context_chunks = filter_context_by_query(context_chunks, norm_query)
# # #
# # #     # Prepare context text
# # #     context_text = "\n".join([c["text"] for c in context_chunks])
# # #     full_context = f"Conversation so far:\n{history_context}\nNew context:\n{context_text}"
# # #
# # #     # Choose model
# # #     if "OpenAI" in model_choice:
# # #         answer = ask_openai(full_context, user_input)
# # #     else:
# # #         answer = ask_ollama(full_context, user_input)
# # #
# # #     # Display answer
# # #     st.markdown(f"**You:** {user_input}")
# # #     st.markdown(f"**Assistant:** {answer}")
# # #
# # #     # Save in conversation history
# # #     st.session_state.conversation_history.append((user_input, answer))
# # #
# # #     # Show sources
# # #     st.subheader("📄 Sources:")
# # #     for src in sorted(set(c["source"] for c in context_chunks)):
# # #         st.write(f"- {src}")
# # # src/faq_ui.py
# #
# # import streamlit as st
# # from faq_assistant import (
# #     normalize_query,
# #     search_index,
# #     ask_openai,
# #     ask_ollama,
# #     load_faiss,
# #     USE_LOCAL,
# #     TOP_K,
# #     MEMORY_TURNS
# # )
# #
# # # ==== Load FAISS & Model ====
# # @st.cache_resource
# # def load_resources():
# #     return load_faiss()
# #
# # model, index, metadata, chunks = load_resources()
# #
# # # ==== Streamlit UI ====
# # st.set_page_config(page_title="BU Graduate Programs FAQ Assistant", page_icon="🎓", layout="wide")
# #
# # st.title("🎓 BU Graduate Programs FAQ Assistant")
# # st.write("Ask me anything about **BU MET graduate programs**.")
# #
# # if "conversation_history" not in st.session_state:
# #     st.session_state.conversation_history = []
# #
# # query = st.text_input("Enter your question:")
# #
# # if query:
# #     norm_query = normalize_query(query)
# #
# #     # Build conversational history
# #     history_context = ""
# #     if st.session_state.conversation_history:
# #         for q, a in st.session_state.conversation_history[-MEMORY_TURNS:]:
# #             history_context += f"User: {q}\nAssistant: {a}\n"
# #
# #     # Retrieve context chunks
# #     context_chunks = search_index(norm_query, model, index, metadata, chunks, top_k=TOP_K)
# #     context_text = "\n\n".join([c["text"] for c in context_chunks])
# #
# #     # Combine context + history
# #     full_context = f"Conversation so far:\n{history_context}\nNew context:\n{context_text}"
# #
# #     # Ask the selected model
# #     if USE_LOCAL:
# #         answer = ask_ollama(full_context, query)
# #     else:
# #         answer = ask_openai(full_context, query)
# #
# #     # Save conversation
# #     st.session_state.conversation_history.append((query, answer))
# #
# #     # Display
# #     st.markdown(f"**You:** {query}")
# #     st.markdown(f"**Assistant:** {answer}")
# #
# #     # Show sources
# #     with st.expander("📄 Sources"):
# #         for src in sorted(set(c["source"] for c in context_chunks)):
# #             st.write(f"- {src}")
# #
# # # Show previous conversation
# # if st.session_state.conversation_history:
# #     st.subheader("💬 Conversation History")
# #     for q, a in st.session_state.conversation_history:
# #         st.markdown(f"**You:** {q}")
# #         st.markdown(f"**Assistant:** {a}")
# # src/faq_ui.py
#
# import streamlit as st
# from faq_assistant import search_index, ask_openai, ask_ollama, normalize_query, load_faiss, load_memory, save_memory, USE_LOCAL, TOP_K, MEMORY_TURNS
#
# # Load resources once
# @st.cache_resource
# def load_resources():
#     return load_faiss()
#
# model, index, metadata, chunks = load_resources()
#
# # Page config
# st.set_page_config(page_title="BU Graduate Programs FAQ Assistant", page_icon="🎓")
# st.title("🎓 BU Graduate Programs FAQ Assistant")
# st.write("Ask me anything about BU MET graduate programs.")
#
# # Conversation memory
# if "conversation_history" not in st.session_state:
#     st.session_state.conversation_history = load_memory()
#
# query = st.text_input("Enter your question:")
#
# if query:
#     norm_query = normalize_query(query)
#
#     # Combined query for better retrieval
#     combined_query = norm_query
#     if st.session_state.conversation_history:
#         last_queries = " ".join([q for q, _ in st.session_state.conversation_history[-MEMORY_TURNS:]])
#         combined_query = f"{norm_query} {last_queries}"
#
#     # Retrieve context
#     context_chunks = search_index(combined_query, model, index, metadata, chunks, top_k=TOP_K)
#     context_text = "\n".join([c["text"] for c in context_chunks])
#
#     # Build conversation history text
#     history_context = ""
#     for q, a in st.session_state.conversation_history[-MEMORY_TURNS:]:
#         history_context += f"User: {q}\nAssistant: {a}\n"
#
#     full_context = f"Conversation so far:\n{history_context}\nNew context:\n{context_text}"
#
#     # Ask model
#     if USE_LOCAL:
#         answer = ask_ollama(full_context, query)
#     else:
#         answer = ask_openai(full_context, query)
#
#     # Display
#     st.write(f"**You:** {query}")
#     st.write(f"**Assistant:** {answer}")
#
#     # Save conversation
#     st.session_state.conversation_history.append((query, answer))
#     save_memory(st.session_state.conversation_history)
#
#     # Show sources
#     with st.expander("📄 Sources"):
#         for src in sorted(set(c["source"] for c in context_chunks)):
#             st.write(f"- {src}")
# src/faq_ui.py

import streamlit as st
from faq_assistant import (
    search_index, ask_openai, ask_ollama,
    normalize_query, load_faiss, load_memory, save_memory
)

# ===== Cached Resource Loader =====
@st.cache_resource
def load_resources():
    return load_faiss()

model, index, metadata, chunks = load_resources()

# ===== Sidebar Controls =====
st.sidebar.title("⚙️ Settings")

model_choice = st.sidebar.radio(
    "Select Model:",
    ["OpenAI", "Ollama"],
    index=0
)

top_k_choice = st.sidebar.slider(
    "Top K Chunks",
    min_value=1, max_value=10,
    value=5, step=1
)

memory_turns_choice = st.sidebar.slider(
    "Conversation Memory (turns)",
    min_value=0, max_value=10,
    value=3, step=1
)

if "USE_LOCAL" not in st.session_state:
    st.session_state.USE_LOCAL = (model_choice == "Ollama")

if "TOP_K" not in st.session_state:
    st.session_state.TOP_K = top_k_choice

if "MEMORY_TURNS" not in st.session_state:
    st.session_state.MEMORY_TURNS = memory_turns_choice

# ===== Page Config =====
st.set_page_config(page_title="BU Graduate Programs FAQ Assistant", page_icon="🎓")
st.title("🎓 BU Graduate Programs FAQ Assistant")

# ===== Conversation Memory =====
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = load_memory()

# ===== Instructions =====
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    1. **Select Model** in the sidebar (OpenAI or Ollama).
    2. **Adjust Retrieval Settings**:
       - **Top K Chunks** → how many text chunks to retrieve for context.
       - **Conversation Memory** → how many past Q&A pairs to include in the prompt.
    3. **Ask Your Question** below.
    4. Press **Enter** or click **Ask** to get your answer.
    """)

# ===== Question Input =====
query = st.text_input("Enter your question:")

if query:
    norm_query = normalize_query(query)

    # Combine with past queries for better retrieval
    combined_query = norm_query
    if st.session_state.conversation_history:
        last_queries = " ".join([q for q, _ in st.session_state.conversation_history[-memory_turns_choice:]])
        combined_query = f"{norm_query} {last_queries}"

    # Retrieve relevant chunks
    context_chunks = search_index(
        combined_query, model, index, metadata, chunks, top_k=top_k_choice
    )
    context_text = "\n".join([c["text"] for c in context_chunks])

    # Build conversation history
    history_context = ""
    for q, a in st.session_state.conversation_history[-memory_turns_choice:]:
        history_context += f"User: {q}\nAssistant: {a}\n"

    # Combine into full context
    full_context = f"Conversation so far:\n{history_context}\nNew context:\n{context_text}"

    # Send to selected model
    if model_choice == "Ollama":
        answer = ask_ollama(full_context, query)
    else:
        answer = ask_openai(full_context, query)

    # ===== Display Chat =====
    st.markdown(f"**You:** {query}")
    st.markdown(f"**Assistant:** {answer}")

    # Save to conversation
    st.session_state.conversation_history.append((query, answer))
    save_memory(st.session_state.conversation_history)

    # ===== Show Sources =====
    with st.expander("📄 Sources"):
        for src in sorted(set(c["source"] for c in context_chunks)):
            st.write(f"- {src}")
