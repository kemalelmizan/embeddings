import numpy as np
import ollama
import time

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embed(texts, model="nomic-embed-text"):
    res = ollama.embed(model=model, input=texts)
    return [np.array(e) for e in res.get("embeddings", [])]

def main():
    docs = [
        "Dune is a science fiction novel by Frank Herbert.",
        "The Hobbit is a fantasy novel by J.R.R. Tolkien.",
        "Neuromancer is a cyberpunk novel by William Gibson.",
        "Pride and Prejudice is a romantic novel by Jane Austen."
    ]
    query = "Who wrote a fantasy novel about hobbits?"

    start_time = time.time()
    doc_embeds, [query_embed] = embed(docs), embed([query])
    scores = [cosine_similarity(query_embed, d) for d in doc_embeds]
    best_idx = int(np.argmax(scores))
    elapsed = time.time() - start_time

    # Estimate token count (simple whitespace split)
    total_tokens = sum(len(doc.split()) for doc in docs) + len(query.split())
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else float('inf')

    print(f"\nQuery: {query}\n")
    print("Similarity scores:")
    for doc, score in zip(docs, scores):
        print(f"  {score:.4f} - {doc}")
    print(f"\nMost similar document (score {scores[best_idx]:.4f}): {docs[best_idx]}")
    print(f"\nTime taken: {elapsed:.4f} seconds for {total_tokens} tokens ({tokens_per_sec:.2f} tokens/sec)")

if __name__ == "__main__":
    main()
