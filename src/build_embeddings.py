# src/build_embeddings.py
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = "data/text"
EMB_DIR = "embeddings"

os.makedirs(EMB_DIR, exist_ok=True)

# Load model
print("🔄 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Load and chunk documents
print("📂 Loading and chunking documents...")
docs = []
metadata = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt"):
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                chunks = chunk_text(text)
                for chunk in chunks:
                    docs.append(chunk)
                    metadata.append(filename)

print(f"✅ Loaded {len(docs)} text chunks from {len(set(metadata))} documents.")

# Generate embeddings
print("🔄 Generating embeddings...")
embeddings = model.encode(docs, show_progress_bar=True)

# Create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# Save index & metadata
faiss.write_index(index, os.path.join(EMB_DIR, "index.faiss"))
np.save(os.path.join(EMB_DIR, "metadata.npy"), np.array(metadata))
np.save(os.path.join(EMB_DIR, "chunks.npy"), np.array(docs))

print("\n✅ Embedding build complete!")
print(f"📦 FAISS index saved to: {os.path.join(EMB_DIR, 'index.faiss')}")
print(f"📦 Metadata saved to: {os.path.join(EMB_DIR, 'metadata.npy')}")
print(f"📦 Chunks saved to: {os.path.join(EMB_DIR, 'chunks.npy')}")
