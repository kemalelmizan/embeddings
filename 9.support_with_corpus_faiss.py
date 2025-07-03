import numpy as np
import faiss
import ollama
import time

def embed(texts, model="nomic-embed-text"):
    res = ollama.embed(model=model, input=texts)
    vecs = np.array(res["embeddings"]).astype('float32')
    faiss.normalize_L2(vecs)
    return vecs

def build_faiss_index(corpus_vectors):
    dim = corpus_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity (after normalization)
    index.add(corpus_vectors)
    return index

def find_best_match(query_vec, index, corpus):
    D, I = index.search(query_vec, 1)  # Top-1
    best_idx = int(I[0][0])
    return corpus[best_idx], float(D[0][0])

def generate_response(query, passage, model="gemma3"):
    prompt = f"""
You are a helpful support assistant.

Customer question: "{query}"

Relevant help article: "{passage}"

Reply to the customer with a short, friendly and informative response based on the article. Do not end with a question.
"""
    res = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return res["message"]["content"].strip()

def main():
    corpus = [
        "Your invoice may increase if your data usage exceeds the quota or if new services are added to your account.",
        "Refunds typically take 5-7 business days to process after approval.",
        "You can reset your password by clicking the 'Forgot password' link on the login page.",
        "To cancel your subscription, go to the billing section in your account settings."
    ]

    query = "Why was my bill higher this month?"

    # Embed corpus and build FAISS index
    start_time = time.time()
    corpus_vecs = embed(corpus)
    index = build_faiss_index(corpus_vecs)

    # Embed query
    query_vec = embed([query])
    match, score = find_best_match(query_vec, index, corpus)
    response = generate_response(query, match)
    elapsed = time.time() - start_time
    total_tokens = sum(len(c.split()) for c in corpus) + len(query.split())
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else float('inf')

    print(f"User Query: {query}")
    print(f"Matched Passage (score {score:.4f}): {match}")
    print("\nAssistant Response:")
    print(response)
    print(f"\nTime taken: {elapsed:.4f} seconds for {total_tokens} tokens ({tokens_per_sec:.2f} tokens/sec)")

if __name__ == "__main__":
    main()
