# # # # # src/faq_assistant.py
# # # # import os
# # # # import re
# # # # import argparse
# # # # import numpy as np
# # # # import faiss
# # # # from sentence_transformers import SentenceTransformer
# # # # from dotenv import load_dotenv
# # # #
# # # # # Load .env variables
# # # # load_dotenv()
# # # #
# # # # # Paths
# # # # EMB_DIR = "embeddings"
# # # # INDEX_FILE = os.path.join(EMB_DIR, "index.faiss")
# # # # META_FILE = os.path.join(EMB_DIR, "metadata.npy")
# # # # CHUNKS_FILE = os.path.join(EMB_DIR, "chunks.npy")
# # # #
# # # # # Models
# # # # DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
# # # # DEFAULT_OLLAMA_MODEL = "mistral"
# # # # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# # # #
# # # # # Alias normalization
# # # # def normalize_query(query: str) -> str:
# # # #     PROGRAM_ALIASES = {
# # # #         r"\bms ada\b": "Master of Science in Applied Data Analytics",
# # # #         r"\bms applied data analytics\b": "Master of Science in Applied Data Analytics",
# # # #         r"\bms cis\b": "Master of Science in Computer Information Systems",
# # # #         r"\bms computer information systems\b": "Master of Science in Computer Information Systems",
# # # #         r"\bms aba\b": "Master of Science in Applied Business Analytics",
# # # #         r"\bms applied business analytics\b": "Master of Science in Applied Business Analytics",
# # # #         r"\bmscj\b": "Master of Science in Criminal Justice",
# # # #     }
# # # #     text = query.lower()
# # # #     for pattern, replacement in PROGRAM_ALIASES.items():
# # # #         text = re.sub(pattern, replacement.lower(), text)
# # # #     return text
# # # #
# # # # # Load FAISS index & embeddings
# # # # def load_faiss():
# # # #     print("🔄 Loading embedding model & FAISS index...")
# # # #     model = SentenceTransformer("all-MiniLM-L6-v2")
# # # #     index = faiss.read_index(INDEX_FILE)
# # # #     metadata = np.load(META_FILE, allow_pickle=True)
# # # #     chunks = np.load(CHUNKS_FILE, allow_pickle=True)
# # # #     return model, index, metadata, chunks
# # # #
# # # # # Search FAISS index
# # # # def search_index(query, model, index, metadata, chunks, top_k=5):
# # # #     query_vec = model.encode([query])
# # # #     distances, indices = index.search(np.array(query_vec), top_k)
# # # #     results = []
# # # #     for i, idx in enumerate(indices[0]):
# # # #         if idx < len(chunks):
# # # #             results.append({
# # # #                 "text": chunks[idx],
# # # #                 "source": metadata[idx],
# # # #                 "score": float(distances[0][i])
# # # #             })
# # # #     return results
# # # #
# # # # # Ask OpenAI
# # # # def ask_openai(context, question, model=DEFAULT_OPENAI_MODEL):
# # # #     from openai import OpenAI
# # # #     client = OpenAI(api_key=OPENAI_API_KEY)
# # # #     prompt = f"""You are a helpful assistant. Use only the context below to answer the question.
# # # #
# # # # Context:
# # # # {context}
# # # #
# # # # Question:
# # # # {question}
# # # #
# # # # Answer:"""
# # # #     print("💬 Sending to OpenAI...")
# # # #     response = client.chat.completions.create(
# # # #         model=model,
# # # #         messages=[{"role": "user", "content": prompt}],
# # # #         temperature=0.2
# # # #     )
# # # #     return response.choices[0].message.content
# # # #
# # # # # Ask Ollama (Local LLM)
# # # # def ask_ollama(context, question, model=DEFAULT_OLLAMA_MODEL):
# # # #     import subprocess
# # # #     prompt = f"""You are a helpful assistant. Use only the context below to answer the question.
# # # #
# # # # Context:
# # # # {context}
# # # #
# # # # Question:
# # # # {question}
# # # #
# # # # Answer:"""
# # # #     print("💬 Sending to Ollama...")
# # # #     result = subprocess.run(
# # # #         ["ollama", "run", model],
# # # #         input=prompt,
# # # #         capture_output=True,
# # # #         text=True
# # # #     )
# # # #     return result.stdout.strip()
# # # #
# # # # # Main interactive loop
# # # # def main():
# # # #     parser = argparse.ArgumentParser()
# # # #     parser.add_argument("--local", action="store_true", help="Use local Ollama model instead of OpenAI")
# # # #     args = parser.parse_args()
# # # #
# # # #     model, index, metadata, chunks = load_faiss()
# # # #
# # # #     while True:
# # # #         query = input("\n❓ Enter your question (or 'exit' to quit): ").strip()
# # # #         if query.lower() == "exit":
# # # #             break
# # # #
# # # #         norm_query = normalize_query(query)
# # # #         results = search_index(norm_query, model, index, metadata, chunks, top_k=5)
# # # #         context_text = "\n\n".join([r["text"] for r in results])
# # # #         retrieved_sources = [r["source"] for r in results]
# # # #
# # # #         if args.local:
# # # #             answer = ask_ollama(context_text, query)
# # # #         else:
# # # #             answer = ask_openai(context_text, query)
# # # #
# # # #         print("\n💡 Answer:")
# # # #         print(answer)
# # # #
# # # #         print("\n📄 Sources:")
# # # #         for src in sorted(set(retrieved_sources)):
# # # #             print(f"- {src}")
# # # #
# # # #         with open("qa_log.txt", "a", encoding="utf-8") as log:
# # # #             log.write(f"Q: {query}\nA: {answer}\nSources: {retrieved_sources}\n{'='*50}\n")
# # # #
# # # # if __name__ == "__main__":
# # # #     main()
# # # # # src/faq_assistant.py
# # # #
# # # # import os
# # # # import re
# # # # import faiss
# # # # import numpy as np
# # # # from sentence_transformers import SentenceTransformer
# # # # from dotenv import load_dotenv
# # # #
# # # # # ==== CONFIG ====
# # # # USE_LOCAL = False  # True = use Ollama, False = use OpenAI
# # # # LOCAL_MODEL = "mistral"
# # # # OPENAI_MODEL = "gpt-4o-mini"
# # # # TOP_K = 5
# # # # MEMORY_TURNS = 3  # Keep last 3 Q&A pairs
# # # #
# # # # # ==== LOAD ENV ====
# # # # load_dotenv()
# # # # if not USE_LOCAL:
# # # #     from openai import OpenAI
# # # #     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# # # #
# # # # # ==== NORMALIZE QUERY ====
# # # # def normalize_query(query: str) -> str:
# # # #     """
# # # #     Cleans and standardizes user queries for better FAISS matching.
# # # #     Expands common abbreviations for BU MET programs.
# # # #     """
# # # #     q = query.strip().lower()
# # # #
# # # #     # Expand common BU MET abbreviations
# # # #     q = re.sub(r"\bms ada\b", "master of science in applied data analytics", q)
# # # #     q = re.sub(r"\bms cis\b", "master of science in computer information systems", q)
# # # #     q = re.sub(r"\bms cs\b", "master of science in computer science", q)
# # # #     q = re.sub(r"\baba\b", "applied business analytics", q)
# # # #
# # # #     # Remove extra punctuation
# # # #     q = re.sub(r"[^\w\s]", " ", q)
# # # #
# # # #     # Collapse multiple spaces
# # # #     q = re.sub(r"\s+", " ", q).strip()
# # # #
# # # #     return q
# # # #
# # # # # ==== LOAD FAISS & MODEL ====
# # # # def load_faiss():
# # # #     model = SentenceTransformer("all-MiniLM-L6-v2")
# # # #     index = faiss.read_index("embeddings/index.faiss")
# # # #     metadata = np.load("embeddings/metadata.npy", allow_pickle=True)
# # # #     chunks = np.load("embeddings/chunks.npy", allow_pickle=True)
# # # #     return model, index, metadata, chunks
# # # #
# # # # model, index, metadata, chunks = load_faiss()
# # # #
# # # # # ==== SEARCH ====
# # # # def search_index(query, model, index, metadata, chunks, top_k=TOP_K):
# # # #     query_emb = model.encode([query])
# # # #     D, I = index.search(np.array(query_emb), top_k)
# # # #     results = []
# # # #     for idx in I[0]:
# # # #         if idx < len(metadata):
# # # #             results.append({"text": chunks[idx], "source": metadata[idx]})
# # # #     return results
# # # #
# # # # # ==== ASK OPENAI ====
# # # # def ask_openai(context, query):
# # # #     messages = [
# # # #         {"role": "system", "content": "You are a helpful assistant for BU MET graduate program FAQs."},
# # # #         {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
# # # #     ]
# # # #     resp = client.chat.completions.create(
# # # #         model=OPENAI_MODEL,
# # # #         messages=messages,
# # # #         temperature=0.2
# # # #     )
# # # #     return resp.choices[0].message.content.strip()
# # # #
# # # # # ==== ASK OLLAMA ====
# # # # def ask_ollama(context, query, model_name=LOCAL_MODEL):
# # # #     import subprocess
# # # #     prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
# # # #     result = subprocess.run(
# # # #         ["ollama", "run", model_name],
# # # #         input=prompt,
# # # #         capture_output=True,
# # # #         text=True
# # # #     )
# # # #     return result.stdout.strip()
# # # #
# # # # # ==== MAIN ====
# # # # def main():
# # # #     print("🔄 Loading embedding model & FAISS index...")
# # # #     conversation_history = []
# # # #
# # # #     while True:
# # # #         query = input("\n❓ Enter your question (or 'exit' to quit): ").strip()
# # # #         if query.lower() in ["exit", "quit"]:
# # # #             break
# # # #
# # # #         # Normalize the query for better retrieval
# # # #         norm_query = normalize_query(query)
# # # #
# # # #         # Build conversational history
# # # #         history_context = ""
# # # #         if conversation_history:
# # # #             for q, a in conversation_history[-MEMORY_TURNS:]:
# # # #                 history_context += f"User: {q}\nAssistant: {a}\n"
# # # #
# # # #         # Retrieve relevant chunks
# # # #         context_chunks = search_index(norm_query, model, index, metadata, chunks, top_k=TOP_K)
# # # #         context_text = "\n".join([c["text"] for c in context_chunks])
# # # #
# # # #         # Combine context + history
# # # #         full_context = f"Conversation so far:\n{history_context}\nNew context:\n{context_text}"
# # # #
# # # #         # Ask model
# # # #         if USE_LOCAL:
# # # #             print("💬 Sending to Ollama...")
# # # #             answer = ask_ollama(full_context, query)
# # # #         else:
# # # #             print("💬 Sending to OpenAI...")
# # # #             answer = ask_openai(full_context, query)
# # # #
# # # #         # Display & save
# # # #         print(f"\n💡 Answer:\n{answer}")
# # # #         conversation_history.append((query, answer))
# # # #
# # # #         # Show sources
# # # #         print("\n📄 Sources:")
# # # #         for src in sorted(set(c["source"] for c in context_chunks)):
# # # #             print(f"- {src}")
# # # #
# # # # if __name__ == "__main__":
# # # #     main()
# # # # # src/faq_assistant.py
# # # #
# # # # import os
# # # # import re
# # # # import faiss
# # # # import numpy as np
# # # # from sentence_transformers import SentenceTransformer
# # # # from dotenv import load_dotenv
# # # #
# # # # # ==== CONFIG ====
# # # # USE_LOCAL = False  # True = use Ollama, False = use OpenAI
# # # # LOCAL_MODEL = "mistral"
# # # # OPENAI_MODEL = "gpt-4o-mini"
# # # # TOP_K = 5
# # # # MEMORY_TURNS = 3  # Keep last 3 Q&A pairs
# # # #
# # # # # ==== LOAD ENV ====
# # # # load_dotenv()
# # # # if not USE_LOCAL:
# # # #     from openai import OpenAI
# # # #     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# # # #
# # # # # ==== NORMALIZE QUERY ====
# # # # def normalize_query(query: str) -> str:
# # # #     """
# # # #     Cleans and standardizes user queries for better FAISS matching.
# # # #     Expands common abbreviations for BU MET programs.
# # # #     """
# # # #     q = query.strip().lower()
# # # #
# # # #     # Expand common BU MET abbreviations
# # # #     q = re.sub(r"\bms ada\b", "master of science in applied data analytics", q)
# # # #     q = re.sub(r"\bms cis\b", "master of science in computer information systems", q)
# # # #     q = re.sub(r"\bms cs\b", "master of science in computer science", q)
# # # #     q = re.sub(r"\baba\b", "applied business analytics", q)
# # # #
# # # #     # Remove extra punctuation
# # # #     q = re.sub(r"[^\w\s]", " ", q)
# # # #
# # # #     # Collapse multiple spaces
# # # #     q = re.sub(r"\s+", " ", q).strip()
# # # #
# # # #     return q
# # # #
# # # # # ==== LOAD FAISS & MODEL ====
# # # # def load_faiss():
# # # #     """Load FAISS index & model with absolute paths."""
# # # #     base_dir = os.path.dirname(os.path.abspath(__file__))
# # # #     embeddings_dir = os.path.join(base_dir, "embeddings")
# # # #
# # # #     index_path = os.path.join(embeddings_dir, "index.faiss")
# # # #     metadata_path = os.path.join(embeddings_dir, "metadata.npy")
# # # #     chunks_path = os.path.join(embeddings_dir, "chunks.npy")
# # # #
# # # #     if not os.path.exists(index_path):
# # # #         raise FileNotFoundError(f"❌ FAISS index not found at {index_path}")
# # # #     if not os.path.exists(metadata_path):
# # # #         raise FileNotFoundError(f"❌ Metadata file not found at {metadata_path}")
# # # #     if not os.path.exists(chunks_path):
# # # #         raise FileNotFoundError(f"❌ Chunks file not found at {chunks_path}")
# # # #
# # # #     model = SentenceTransformer("all-MiniLM-L6-v2")
# # # #     index = faiss.read_index(index_path)
# # # #     metadata = np.load(metadata_path, allow_pickle=True)
# # # #     chunks = np.load(chunks_path, allow_pickle=True)
# # # #     return model, index, metadata, chunks
# # # #
# # # # model, index, metadata, chunks = load_faiss()
# # # #
# # # # # ==== SEARCH ====
# # # # def search_index(query, model, index, metadata, chunks, top_k=TOP_K):
# # # #     query_emb = model.encode([query])
# # # #     D, I = index.search(np.array(query_emb), top_k)
# # # #     results = []
# # # #     for idx in I[0]:
# # # #         if idx < len(metadata):
# # # #             results.append({"text": chunks[idx], "source": metadata[idx]})
# # # #     return results
# # # #
# # # # # ==== ASK OPENAI ====
# # # # def ask_openai(context, query):
# # # #     messages = [
# # # #         {"role": "system", "content": "You are a helpful assistant for BU MET graduate program FAQs."},
# # # #         {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
# # # #     ]
# # # #     resp = client.chat.completions.create(
# # # #         model=OPENAI_MODEL,
# # # #         messages=messages,
# # # #         temperature=0.2
# # # #     )
# # # #     return resp.choices[0].message.content.strip()
# # # #
# # # # # ==== ASK OLLAMA ====
# # # # def ask_ollama(context, query, model_name=LOCAL_MODEL):
# # # #     import subprocess
# # # #     prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
# # # #     result = subprocess.run(
# # # #         ["ollama", "run", model_name],
# # # #         input=prompt,
# # # #         capture_output=True,
# # # #         text=True
# # # #     )
# # # #     return result.stdout.strip()
# # # #
# # # # # ==== MAIN ====
# # # # def main():
# # # #     print("🔄 Loading embedding model & FAISS index...")
# # # #     conversation_history = []
# # # #
# # # #     while True:
# # # #         query = input("\n❓ Enter your question (or 'exit' to quit): ").strip()
# # # #         if query.lower() in ["exit", "quit"]:
# # # #             break
# # # #
# # # #         # Normalize the query for better retrieval
# # # #         norm_query = normalize_query(query)
# # # #
# # # #         # Build conversational history
# # # #         history_context = ""
# # # #         if conversation_history:
# # # #             for q, a in conversation_history[-MEMORY_TURNS:]:
# # # #                 history_context += f"User: {q}\nAssistant: {a}\n"
# # # #
# # # #         # Retrieve relevant chunks
# # # #         context_chunks = search_index(norm_query, model, index, metadata, chunks, top_k=TOP_K)
# # # #         context_text = "\n".join([c["text"] for c in context_chunks])
# # # #
# # # #         # Combine context + history
# # # #         full_context = f"Conversation so far:\n{history_context}\nNew context:\n{context_text}"
# # # #
# # # #         # Ask model
# # # #         if USE_LOCAL:
# # # #             print("💬 Sending to Ollama...")
# # # #             answer = ask_ollama(full_context, query)
# # # #         else:
# # # #             print("💬 Sending to OpenAI...")
# # # #             answer = ask_openai(full_context, query)
# # # #
# # # #         # Display & save
# # # #         print(f"\n💡 Answer:\n{answer}")
# # # #         conversation_history.append((query, answer))
# # # #
# # # #         # Show sources
# # # #         print("\n📄 Sources:")
# # # #         for src in sorted(set(c["source"] for c in context_chunks)):
# # # #             print(f"- {src}")
# # # #
# # # # if __name__ == "__main__":
# # # #     main()
# # #
# # # # src/faq_assistant.py
# # #
# # # import os
# # # import re
# # # import faiss
# # # import numpy as np
# # # from sentence_transformers import SentenceTransformer
# # # from dotenv import load_dotenv
# # #
# # # # ==== CONFIG ====
# # # USE_LOCAL = False  # True = use Ollama, False = use OpenAI
# # # LOCAL_MODEL = "mistral"  # Ollama model
# # # OPENAI_MODEL = "gpt-4o-mini"
# # # TOP_K = 5
# # # MEMORY_TURNS = 3  # Keep last N Q&A pairs in memory
# # #
# # # # ==== LOAD ENV ====
# # # load_dotenv()
# # # if not USE_LOCAL:
# # #     from openai import OpenAI
# # #     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# # #
# # # # ==== NORMALIZE QUERY ====
# # # def normalize_query(query: str) -> str:
# # #     """
# # #     Cleans and standardizes user queries for better FAISS matching.
# # #     Expands common abbreviations for BU MET programs.
# # #     """
# # #     q = query.strip().lower()
# # #
# # #     # Expand BU MET abbreviations
# # #     q = re.sub(r"\bms ada\b", "master of science in applied data analytics", q)
# # #     q = re.sub(r"\bms cis\b", "master of science in computer information systems", q)
# # #     q = re.sub(r"\bms cs\b", "master of science in computer science", q)
# # #     q = re.sub(r"\baba\b", "applied business analytics", q)
# # #
# # #     # Remove extra punctuation
# # #     q = re.sub(r"[^\w\s]", " ", q)
# # #
# # #     # Collapse multiple spaces
# # #     q = re.sub(r"\s+", " ", q).strip()
# # #
# # #     return q
# # #
# # # # ==== LOAD FAISS & MODEL ====
# # # def load_faiss():
# # #     """Load FAISS index & model with absolute paths."""
# # #     base_dir = os.path.dirname(os.path.abspath(__file__))
# # #     embeddings_dir = os.path.join(base_dir, "embeddings")  # inside src/
# # #
# # #     index_path = os.path.join(embeddings_dir, "index.faiss")
# # #     metadata_path = os.path.join(embeddings_dir, "metadata.npy")
# # #     chunks_path = os.path.join(embeddings_dir, "chunks.npy")
# # #
# # #     if not os.path.exists(index_path):
# # #         raise FileNotFoundError(f"❌ FAISS index not found at {index_path}")
# # #     if not os.path.exists(metadata_path):
# # #         raise FileNotFoundError(f"❌ Metadata file not found at {metadata_path}")
# # #     if not os.path.exists(chunks_path):
# # #         raise FileNotFoundError(f"❌ Chunks file not found at {chunks_path}")
# # #
# # #     model = SentenceTransformer("all-MiniLM-L6-v2")
# # #     index = faiss.read_index(index_path)
# # #     metadata = np.load(metadata_path, allow_pickle=True)
# # #     chunks = np.load(chunks_path, allow_pickle=True)
# # #     return model, index, metadata, chunks
# # #
# # # # Load once at start
# # # model, index, metadata, chunks = load_faiss()
# # #
# # # # ==== SEARCH ====
# # # def search_index(query, model, index, metadata, chunks, top_k=TOP_K):
# # #     """Search FAISS index for most relevant chunks."""
# # #     query_emb = model.encode([query])
# # #     D, I = index.search(np.array(query_emb), top_k)
# # #     results = []
# # #     for idx in I[0]:
# # #         if idx < len(metadata):
# # #             results.append({"text": chunks[idx], "source": metadata[idx]})
# # #     return results
# # #
# # # # ==== ASK OPENAI ====
# # # def ask_openai(context, query):
# # #     messages = [
# # #         {"role": "system", "content": "You are a helpful assistant for BU MET graduate program FAQs."},
# # #         {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
# # #     ]
# # #     resp = client.chat.completions.create(
# # #         model=OPENAI_MODEL,
# # #         messages=messages,
# # #         temperature=0.2
# # #     )
# # #     return resp.choices[0].message.content.strip()
# # #
# # # # ==== ASK OLLAMA ====
# # # def ask_ollama(context, query, model_name=LOCAL_MODEL):
# # #     import subprocess
# # #     prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
# # #     result = subprocess.run(
# # #         ["ollama", "run", model_name],
# # #         input=prompt,
# # #         capture_output=True,
# # #         text=True
# # #     )
# # #     return result.stdout.strip()
# # #
# # # # ==== MAIN ====
# # # def main():
# # #     print("🔄 Loading embedding model & FAISS index...")
# # #     conversation_history = []
# # #
# # #     while True:
# # #         query = input("\n❓ Enter your question (or 'exit' to quit): ").strip()
# # #         if query.lower() in ["exit", "quit"]:
# # #             break
# # #
# # #         # Normalize query for better matching
# # #         norm_query = normalize_query(query)
# # #
# # #         # Build conversational memory context
# # #         history_context = ""
# # #         if conversation_history:
# # #             for q, a in conversation_history[-MEMORY_TURNS:]:
# # #                 history_context += f"User: {q}\nAssistant: {a}\n"
# # #
# # #         # Retrieve relevant chunks
# # #         context_chunks = search_index(norm_query, model, index, metadata, chunks, top_k=TOP_K)
# # #         context_text = "\n".join([c["text"] for c in context_chunks])
# # #
# # #         # Combine with history
# # #         full_context = f"Conversation so far:\n{history_context}\nNew context:\n{context_text}"
# # #
# # #         # Ask model
# # #         if USE_LOCAL:
# # #             print("💬 Sending to Ollama...")
# # #             answer = ask_ollama(full_context, query)
# # #         else:
# # #             print("💬 Sending to OpenAI...")
# # #             answer = ask_openai(full_context, query)
# # #
# # #         # Display & save answer
# # #         print(f"\n💡 Answer:\n{answer}")
# # #         conversation_history.append((query, answer))
# # #
# # #         # Show sources
# # #         print("\n📄 Sources:")
# # #         for src in sorted(set(c["source"] for c in context_chunks)):
# # #             print(f"- {src}")
# # #
# # # if __name__ == "__main__":
# # #     main()
# # # src/faq_assistant.py
# #
# # import os
# # import re
# # import faiss
# # import numpy as np
# # from sentence_transformers import SentenceTransformer
# # from dotenv import load_dotenv
# #
# # # ==== CONFIG ====
# # USE_LOCAL = False  # True = use Ollama, False = use OpenAI
# # LOCAL_MODEL = "mistral"
# # OPENAI_MODEL = "gpt-4o-mini"
# # TOP_K = 5
# # MEMORY_TURNS = 3  # Keep last 3 Q&A pairs
# #
# # # ==== LOAD ENV ====
# # load_dotenv()
# # if not USE_LOCAL:
# #     from openai import OpenAI
# #     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# #
# # # ==== NORMALIZE QUERY ====
# # def normalize_query(query: str) -> str:
# #     """
# #     Cleans and standardizes user queries for better FAISS matching.
# #     Expands common abbreviations for BU MET programs.
# #     """
# #     q = query.strip().lower()
# #
# #     # Expand common BU MET abbreviations
# #     q = re.sub(r"\bms ada\b", "master of science in applied data analytics", q)
# #     q = re.sub(r"\bms cis\b", "master of science in computer information systems", q)
# #     q = re.sub(r"\bms cs\b", "master of science in computer science", q)
# #     q = re.sub(r"\baba\b", "applied business analytics", q)
# #     q = re.sub(r"\bmscj\b", "master of science in criminal justice", q)
# #
# #     # Remove extra punctuation
# #     q = re.sub(r"[^\w\s]", " ", q)
# #
# #     # Collapse multiple spaces
# #     q = re.sub(r"\s+", " ", q).strip()
# #
# #     return q
# #
# # # ==== LOAD FAISS & MODEL ====
# # def load_faiss():
# #     """Load FAISS index & model with absolute paths."""
# #     base_dir = os.path.dirname(os.path.abspath(__file__))
# #     embeddings_dir = os.path.join(base_dir, "embeddings")
# #
# #     index_path = os.path.join(embeddings_dir, "index.faiss")
# #     metadata_path = os.path.join(embeddings_dir, "metadata.npy")
# #     chunks_path = os.path.join(embeddings_dir, "chunks.npy")
# #
# #     if not os.path.exists(index_path):
# #         raise FileNotFoundError(f"❌ FAISS index not found at {index_path}")
# #     if not os.path.exists(metadata_path):
# #         raise FileNotFoundError(f"❌ Metadata file not found at {metadata_path}")
# #     if not os.path.exists(chunks_path):
# #         raise FileNotFoundError(f"❌ Chunks file not found at {chunks_path}")
# #
# #     model = SentenceTransformer("all-MiniLM-L6-v2")
# #     index = faiss.read_index(index_path)
# #     metadata = np.load(metadata_path, allow_pickle=True)
# #     chunks = np.load(chunks_path, allow_pickle=True)
# #     return model, index, metadata, chunks
# #
# # model, index, metadata, chunks = load_faiss()
# #
# # # ==== SEARCH ====
# # def search_index(query, model, index, metadata, chunks, top_k=TOP_K):
# #     """
# #     Search FAISS index and return relevant text chunks + sources.
# #     """
# #     query_emb = model.encode([query])
# #     D, I = index.search(np.array(query_emb), top_k)
# #     results = []
# #     for idx in I[0]:
# #         if idx < len(metadata):
# #             results.append({"text": chunks[idx], "source": metadata[idx]})
# #     return results
# #
# # # ==== FILTER CONTEXT BY QUERY ====
# # def filter_context_by_query(context_chunks, query):
# #     """
# #     Filters retrieved chunks to keep only those related to the program or topic in the query.
# #     This helps avoid mixing details from unrelated programs.
# #     """
# #     query_keywords = set(query.lower().split())
# #     filtered = []
# #     for chunk in context_chunks:
# #         if any(keyword in chunk["text"].lower() for keyword in query_keywords):
# #             filtered.append(chunk)
# #     return filtered if filtered else context_chunks  # fallback to all if filter too strict
# #
# # # ==== ASK OPENAI ====
# # def ask_openai(context, query):
# #     messages = [
# #         {
# #             "role": "system",
# #             "content": (
# #                 "You are a helpful assistant for BU MET graduate program FAQs. "
# #                 "Only use the provided context to answer. "
# #                 "If the context does not contain relevant information, say: "
# #                 "'I don't have that information in the provided documents.'"
# #             )
# #         },
# #         {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
# #     ]
# #     resp = client.chat.completions.create(
# #         model=OPENAI_MODEL,
# #         messages=messages,
# #         temperature=0.2
# #     )
# #     return resp.choices[0].message.content.strip()
# #
# # # ==== ASK OLLAMA ====
# # def ask_ollama(context, query, model_name=LOCAL_MODEL):
# #     import subprocess
# #     prompt = (
# #         "You are a helpful assistant for BU MET graduate program FAQs. "
# #         "Only use the provided context to answer. "
# #         "If the context does not contain relevant information, say: "
# #         "'I don't have that information in the provided documents.'\n\n"
# #         f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
# #     )
# #     result = subprocess.run(
# #         ["ollama", "run", model_name],
# #         input=prompt,
# #         capture_output=True,
# #         text=True
# #     )
# #     return result.stdout.strip()
# #
# # # ==== MAIN ====
# # def main():
# #     print("🔄 Loading embedding model & FAISS index...")
# #     conversation_history = []
# #
# #     while True:
# #         query = input("\n❓ Enter your question (or 'exit' to quit): ").strip()
# #         if query.lower() in ["exit", "quit"]:
# #             break
# #
# #         # Normalize query
# #         norm_query = normalize_query(query)
# #
# #         # Build conversational memory
# #         history_context = ""
# #         if conversation_history:
# #             for q, a in conversation_history[-MEMORY_TURNS:]:
# #                 history_context += f"User: {q}\nAssistant: {a}\n"
# #
# #         # Search index
# #         context_chunks = search_index(norm_query, model, index, metadata, chunks, top_k=TOP_K)
# #
# #         # Filter context for accuracy
# #         context_chunks = filter_context_by_query(context_chunks, norm_query)
# #
# #         # Create context text
# #         context_text = "\n".join([c["text"] for c in context_chunks])
# #         full_context = f"Conversation so far:\n{history_context}\nNew context:\n{context_text}"
# #
# #         # Ask selected model
# #         if USE_LOCAL:
# #             print("💬 Sending to Ollama...")
# #             answer = ask_ollama(full_context, query)
# #         else:
# #             print("💬 Sending to OpenAI...")
# #             answer = ask_openai(full_context, query)
# #
# #         # Display & store in memory
# #         print(f"\n💡 Answer:\n{answer}")
# #         conversation_history.append((query, answer))
# #
# #         # Show sources
# #         print("\n📄 Sources:")
# #         for src in sorted(set(c["source"] for c in context_chunks)):
# #             print(f"- {src}")
# #
# # if __name__ == "__main__":
# #     main()
# # src/faq_assistant.py
#
# import os
# import re
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
#
# # ==== CONFIG ====
# USE_LOCAL = False  # True = use Ollama, False = use OpenAI
# LOCAL_MODEL = "mistral"
# OPENAI_MODEL = "gpt-4o-mini"
# TOP_K = 15  # Increased from 5 to capture more relevant chunks
# MEMORY_TURNS = 3  # Keep last 3 Q&A pairs
#
# # ==== LOAD ENV ====
# load_dotenv()
# if not USE_LOCAL:
#     from openai import OpenAI
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#
# # ==== NORMALIZE QUERY ====
# def normalize_query(query: str) -> str:
#     """
#     Cleans and standardizes user queries for better FAISS matching.
#     Expands common abbreviations for BU MET programs.
#     """
#     q = query.strip().lower()
#
#     # Expand common BU MET abbreviations
#     q = re.sub(r"\bms ada\b", "master of science in applied data analytics", q)
#     q = re.sub(r"\bms cis\b", "master of science in computer information systems", q)
#     q = re.sub(r"\bms cs\b", "master of science in computer science", q)
#     q = re.sub(r"\baba\b", "applied business analytics", q)
#
#     # Remove extra punctuation
#     q = re.sub(r"[^\w\s]", " ", q)
#
#     # Collapse multiple spaces
#     q = re.sub(r"\s+", " ", q).strip()
#
#     return q
#
# # ==== LOAD FAISS & MODEL ====
# def load_faiss():
#     """Load FAISS index & model with absolute paths."""
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     embeddings_dir = os.path.join(base_dir, "embeddings")
#
#     index_path = os.path.join(embeddings_dir, "index.faiss")
#     metadata_path = os.path.join(embeddings_dir, "metadata.npy")
#     chunks_path = os.path.join(embeddings_dir, "chunks.npy")
#
#     if not os.path.exists(index_path):
#         raise FileNotFoundError(f"❌ FAISS index not found at {index_path}")
#     if not os.path.exists(metadata_path):
#         raise FileNotFoundError(f"❌ Metadata file not found at {metadata_path}")
#     if not os.path.exists(chunks_path):
#         raise FileNotFoundError(f"❌ Chunks file not found at {chunks_path}")
#
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     index = faiss.read_index(index_path)
#     metadata = np.load(metadata_path, allow_pickle=True)
#     chunks = np.load(chunks_path, allow_pickle=True)
#     return model, index, metadata, chunks
#
# model, index, metadata, chunks = load_faiss()
#
# # ==== SEARCH ====
# def search_index(query, model, index, metadata, chunks, top_k=TOP_K):
#     query_emb = model.encode([query])
#     D, I = index.search(np.array(query_emb), top_k)
#     results = []
#     for idx in I[0]:
#         if idx < len(metadata):
#             results.append({"text": chunks[idx], "source": metadata[idx]})
#     return results
#
# # ==== ASK OPENAI ====
# def ask_openai(context, query):
#     messages = [
#         {"role": "system", "content": (
#             "You are a helpful assistant for BU MET graduate program FAQs.\n"
#             "You MUST use the provided context to answer the question.\n"
#             "If the context contains partial information, use it and your best interpretation to provide a helpful answer.\n"
#             "Only say 'I could not find details in the provided context.' if there is absolutely nothing relevant."
#         )},
#         {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
#     ]
#     resp = client.chat.completions.create(
#         model=OPENAI_MODEL,
#         messages=messages,
#         temperature=0.2
#     )
#     return resp.choices[0].message.content.strip()
#
# # ==== ASK OLLAMA ====
# def ask_ollama(context, query, model_name=LOCAL_MODEL):
#     import subprocess
#     prompt = (
#         "You are a helpful assistant for BU MET graduate program FAQs.\n"
#         "You MUST use the provided context to answer the question.\n"
#         "If the context contains partial information, use it and your best interpretation to provide a helpful answer.\n"
#         "Only say 'I could not find details in the provided context.' if there is absolutely nothing relevant.\n\n"
#         f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
#     )
#     result = subprocess.run(
#         ["ollama", "run", model_name],
#         input=prompt,
#         capture_output=True,
#         text=True
#     )
#     return result.stdout.strip()
#
# # ==== MAIN ====
# def main():
#     print("🔄 Loading embedding model & FAISS index...")
#     conversation_history = []
#
#     while True:
#         query = input("\n❓ Enter your question (or 'exit' to quit): ").strip()
#         if query.lower() in ["exit", "quit"]:
#             break
#
#         # Normalize the query for better retrieval
#         norm_query = normalize_query(query)
#
#         # Build conversational history
#         history_context = ""
#         if conversation_history:
#             for q, a in conversation_history[-MEMORY_TURNS:]:
#                 history_context += f"User: {q}\nAssistant: {a}\n"
#
#         # Retrieve relevant chunks
#         context_chunks = search_index(norm_query, model, index, metadata, chunks, top_k=TOP_K)
#         context_text = "\n\n".join([c["text"] for c in context_chunks])  # Keep paragraph spacing
#
#         # Combine context + history
#         full_context = f"Conversation so far:\n{history_context}\nNew context:\n{context_text}"
#
#         # Ask model
#         if USE_LOCAL:
#             print("💬 Sending to Ollama...")
#             answer = ask_ollama(full_context, query)
#         else:
#             print("💬 Sending to OpenAI...")
#             answer = ask_openai(full_context, query)
#
#         # Display & save
#         print(f"\n💡 Answer:\n{answer}")
#         conversation_history.append((query, answer))
#
#         # Show sources
#         print("\n📄 Sources:")
#         for src in sorted(set(c["source"] for c in context_chunks)):
#             print(f"- {src}")
#
# if __name__ == "__main__":
#     main()
# src/faq_assistant.py

import os
import re
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ==== CONFIG ====
USE_LOCAL = False  # True = use Ollama, False = use OpenAI
LOCAL_MODEL = "mistral"
OPENAI_MODEL = "gpt-4o-mini"
TOP_K = 5
MEMORY_TURNS = 3  # Keep last 3 Q&A pairs
MEMORY_FILE = "conversation_memory.json"

# ==== LOAD ENV ====
load_dotenv()
# if not USE_LOCAL:
#     from openai import OpenAI
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Try Streamlit secrets first, then environment variable
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not api_key:
    raise ValueError("OpenAI API key not found. Set it in Streamlit Secrets or as an environment variable.")

client = OpenAI(api_key=api_key)


# ==== MEMORY FUNCTIONS ====
def save_memory(history):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f)

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ==== NORMALIZE QUERY ====
def normalize_query(query: str) -> str:
    q = query.strip().lower()
    q = re.sub(r"\bms ada\b", "master of science in applied data analytics", q)
    q = re.sub(r"\bms cis\b", "master of science in computer information systems", q)
    q = re.sub(r"\bms cs\b", "master of science in computer science", q)
    q = re.sub(r"\baba\b", "applied business analytics", q)
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

# ==== LOAD FAISS & MODEL ====
def load_faiss():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(base_dir, "embeddings")
    index_path = os.path.join(embeddings_dir, "index.faiss")
    metadata_path = os.path.join(embeddings_dir, "metadata.npy")
    chunks_path = os.path.join(embeddings_dir, "chunks.npy")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"❌ FAISS index not found at {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"❌ Metadata file not found at {metadata_path}")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"❌ Chunks file not found at {chunks_path}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(index_path)
    metadata = np.load(metadata_path, allow_pickle=True)
    chunks = np.load(chunks_path, allow_pickle=True)
    return model, index, metadata, chunks

model, index, metadata, chunks = load_faiss()

# ==== SEARCH ====
def search_index(query, model, index, metadata, chunks, top_k=TOP_K):
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), top_k)
    results = []
    for idx in I[0]:
        if idx < len(metadata):
            results.append({"text": chunks[idx], "source": metadata[idx]})
    return results

# ==== ASK OPENAI ====
def ask_openai(context, query):
    messages = [
        {"role": "system", "content": "You are a helpful assistant for BU MET graduate program FAQs."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

# ==== ASK OLLAMA ====
def ask_ollama(context, query, model_name=LOCAL_MODEL):
    import subprocess
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt,
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

# ==== MAIN ====
def main():
    print("🔄 Loading embedding model & FAISS index...")
    conversation_history = load_memory()

    while True:
        query = input("\n❓ Enter your question (or 'exit' to quit): ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        norm_query = normalize_query(query)

        # Combined query for better retrieval
        combined_query = norm_query
        if conversation_history:
            last_queries = " ".join([q for q, _ in conversation_history[-MEMORY_TURNS:]])
            combined_query = f"{norm_query} {last_queries}"

        # Retrieve relevant chunks
        context_chunks = search_index(combined_query, model, index, metadata, chunks, top_k=TOP_K)
        context_text = "\n".join([c["text"] for c in context_chunks])

        # Build conversational context
        history_context = ""
        for q, a in conversation_history[-MEMORY_TURNS:]:
            history_context += f"User: {q}\nAssistant: {a}\n"

        full_context = f"Conversation so far:\n{history_context}\nNew context:\n{context_text}"

        # Ask LLM
        if USE_LOCAL:
            print("💬 Sending to Ollama...")
            answer = ask_ollama(full_context, query)
        else:
            print("💬 Sending to OpenAI...")
            answer = ask_openai(full_context, query)

        # Display & save
        print(f"\n💡 Answer:\n{answer}")
        conversation_history.append((query, answer))
        save_memory(conversation_history)

        print("\n📄 Sources:")
        for src in sorted(set(c["source"] for c in context_chunks)):
            print(f"- {src}")

if __name__ == "__main__":
    main()
