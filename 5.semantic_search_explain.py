import numpy as np
import ollama
import time

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embed(texts, model="nomic-embed-text"):
    res = ollama.embed(model=model, input=texts)
    return np.array(res["embeddings"])

def find_most_similar(query, documents):
    all_embeddings = embed([query] + documents)
    query_vec = all_embeddings[0]
    doc_vecs = all_embeddings[1:]

    similarities = [cosine_similarity(query_vec, doc_vec) for doc_vec in doc_vecs]
    best_idx = int(np.argmax(similarities))
    return best_idx, similarities

def explain_match(query, document, model="gemma3"):
    prompt = f"""
Explain why the following document is relevant to the user's query.

Query: "{query}"

Document: "{document}"

Explanation:
"""
    res = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return res["message"]["content"].strip()

def main():
    documents = [
        "Dune is a science fiction novel set on the desert planet Arrakis, focusing on politics, religion, and ecology.",
        "The Hobbit is a fantasy story by J.R.R. Tolkien about a hobbit's adventure to reclaim a treasure from a dragon.",
        "Neuromancer is a cyberpunk novel involving AI, hacking, and a dystopian future.",
        "Pride and Prejudice is a romantic novel set in 19th-century England, dealing with social class and love.",
    ]

    query = "What book features a dragon and an epic journey?"

    print(f"\U0001F50E Query: {query}\n")

    start_time = time.time()
    best_idx, similarities = find_most_similar(query, documents)
    elapsed = time.time() - start_time
    best_doc = documents[best_idx]
    total_tokens = sum(len(d.split()) for d in documents) + len(query.split())
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else float('inf')

    print("\U0001F4C4 Most Similar Document:")
    print(f"  {best_doc}")
    print(f"  (Similarity Score: {similarities[best_idx]:.4f})")
    print(f"\nTime taken: {elapsed:.4f} seconds for {total_tokens} tokens ({tokens_per_sec:.2f} tokens/sec)")

    explanation = explain_match(query, best_doc)
    print("\n💬 Gemma's Explanation:\n", explanation)

if __name__ == "__main__":
    main()
