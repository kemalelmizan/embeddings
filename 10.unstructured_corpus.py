import numpy as np
import faiss
import ollama
import time

def embed(texts, model="nomic-embed-text"):
    res = ollama.embed(model=model, input=texts)
    vecs = np.array(res["embeddings"]).astype('float32')
    faiss.normalize_L2(vecs)
    return vecs

def build_index(corpus_vectors):
    index = faiss.IndexFlatIP(corpus_vectors.shape[1])
    index.add(corpus_vectors)
    return index

def retrieve(query, index, query_vec, corpus):
    D, I = index.search(query_vec, 1)
    return corpus[I[0][0]], D[0][0]

def explain_with_gemma(query, passage, model="gemma3"):
    prompt = f"""
You are an AI system support assistant.

User question: "{query}"

Log evidence: "{passage}"

Explain briefly the likely cause or helpful insight to a non-technical user.
"""
    res = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return res["message"]["content"].strip()

def main():
    with open("logs.txt", "r") as f:
        raw_logs = [line.strip() for line in f if line.strip()]

    query = "Why are customers complaining about failed checkouts?"

    start_time = time.time()
    corpus_vecs = embed(raw_logs)
    index = build_index(corpus_vecs)
    query_vec = embed([query])
    match, score = retrieve(query, index, query_vec, raw_logs)
    response = explain_with_gemma(query, match)
    elapsed = time.time() - start_time
    total_tokens = sum(len(l.split()) for l in raw_logs) + len(query.split())
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else float('inf')

    print(f"User Query: {query}")
    print(f"\nMatched Log (score {score:.4f}): {match}")
    print("\nAI Explanation:")
    print(response)
    print(f"\nTime taken: {elapsed:.4f} seconds for {total_tokens} tokens ({tokens_per_sec:.2f} tokens/sec)")

if __name__ == "__main__":
    main()
