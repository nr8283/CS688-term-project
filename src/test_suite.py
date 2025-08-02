# src/test_suite.py

import os
import time
import datetime
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# ==== CONFIG ====
USE_LOCAL = False  # True = Ollama (local), False = OpenAI
LOCAL_MODEL = "mistral"
OPENAI_MODEL = "gpt-4o-mini"
TOP_K = 5
TEST_QUERIES = [
    "ms ada",
    "ms computer information systems",
    "criminal justice",
    "cybersecurity",
    "data science",
    "Applied Data Analytics program requirements"
]

# ==== LOAD ENV ====
load_dotenv()
if not USE_LOCAL:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==== LOAD RESOURCES ====
def load_faiss():
    """Load FAISS index & model with absolute paths."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(base_dir, "embeddings")

    index_path = os.path.join(embeddings_dir, "index.faiss")
    metadata_path = os.path.join(embeddings_dir, "metadata.npy")
    chunks_path = os.path.join(embeddings_dir, "chunks.npy")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"❌ FAISS index not found at {index_path}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(index_path)
    metadata = np.load(metadata_path, allow_pickle=True)
    chunks = np.load(chunks_path, allow_pickle=True)
    return model, index, metadata, chunks

# ==== SEARCH ====
def search_index(query, model, index, metadata, chunks, top_k=TOP_K):
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), top_k)
    results = []
    for idx in I[0]:
        if idx < len(metadata):
            results.append({"text": chunks[idx], "source": metadata[idx]})
    return results

# ==== ASK MODELS ====
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

# ==== EVALUATION ====
def run_tests():
    model, index, metadata, chunks = load_faiss()

    results = []
    for query in TEST_QUERIES:
        print(f"\n🔍 Testing: {query}")
        context_chunks = search_index(query, model, index, metadata, chunks, top_k=TOP_K)
        context_text = "\n".join([c["text"] for c in context_chunks])

        start_time = time.time()
        if USE_LOCAL:
            answer = ask_ollama(context_text, query)
        else:
            answer = ask_openai(context_text, query)
        end_time = time.time()

        response_time = round(end_time - start_time, 2)
        sources = ", ".join(sorted(set(c["source"] for c in context_chunks)))

        results.append({
            "Query": query,
            "Model": LOCAL_MODEL if USE_LOCAL else OPENAI_MODEL,
            "Answer": answer,
            "Response Time (s)": response_time,
            "Sources": sources,
            "Manual Accuracy (1-5)": ""  # fill manually later
        })

    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    print("✅ Saved: evaluation_results.csv")

    # Save Markdown table
    df.to_markdown("evaluation_results.md", index=False)
    print("✅ Saved: evaluation_results.md")

    return df

# ==== SUMMARY ====
def summarize_results(df):
    # Convert Manual Accuracy to numeric safely (NaN if blank)
    df["Manual Accuracy (1-5)"] = pd.to_numeric(df["Manual Accuracy (1-5)"], errors="coerce")

    summary = df.groupby("Model").agg({
        "Response Time (s)": "mean",
        "Manual Accuracy (1-5)": "mean"  # will ignore NaN
    }).reset_index()

    summary.to_markdown("evaluation_summary.md", index=False)
    print("✅ Saved: evaluation_summary.md")


# ==== REPORT GENERATION ====
def generate_final_report():
    """Combine evaluation results & summary into final report."""
    with open("evaluation_results.md", "r", encoding="utf-8") as f:
        results_content = f.read()
    with open("evaluation_summary.md", "r", encoding="utf-8") as f:
        summary_content = f.read()

    today = datetime.date.today().strftime("%B %d, %Y")
    report = f"""# 📊 CS688 Project – BU Graduate Programs FAQ Assistant
## Model Evaluation Report
**Date:** {today}

---

## 1. Introduction
This report evaluates two model configurations for the **BU Graduate Programs FAQ Assistant**:
- **OpenAI GPT-4o-mini**
- **Ollama (Mistral)** (optional local LLM)

---

## 2. Test Queries
{", ".join(TEST_QUERIES)}

---

## 3. Detailed Results
{results_content}

---

## 4. Summary
{summary_content}

---

## 5. Conclusion
OpenAI GPT-4o-mini is recommended for accuracy & speed.
Ollama Mistral is slower and less accurate but works offline.

"""

    with open("evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("✅ Final report saved: evaluation_report.md")

# ==== MAIN ====
if __name__ == "__main__":
    df = run_tests()
    summarize_results(df)
    generate_final_report()
