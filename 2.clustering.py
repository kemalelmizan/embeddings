import numpy as np
from sklearn.cluster import KMeans
import ollama
import time

def embed(texts, model="nomic-embed-text"):
    res = ollama.embed(model=model, input=texts)
    return np.array(res.get("embeddings", []))

def cluster_texts(texts, n_clusters=2):
    embeddings = embed(texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    clusters = {i: [] for i in range(n_clusters)}
    for text, label in zip(texts, labels):
        clusters[label].append(text)

    return clusters

def main():
    texts = [
        "Dune is a science fiction novel by Frank Herbert.",
        "Neuromancer is a cyberpunk novel by William Gibson.",
        "The Hobbit is a fantasy novel by J.R.R. Tolkien.",
        "Pride and Prejudice is a romantic novel by Jane Austen.",
        "Hyperion is a science fiction novel by Dan Simmons.",
        "Emma is another romantic novel by Jane Austen.",
    ]

    start_time = time.time()
    clustered = cluster_texts(texts, n_clusters=3)
    elapsed = time.time() - start_time
    total_tokens = sum(len(t.split()) for t in texts)
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else float('inf')

    for cluster_id, items in clustered.items():
        print(f"\nCluster {cluster_id + 1}:")
        for item in items:
            print(f"  - {item}")
    print(f"\nTime taken: {elapsed:.4f} seconds for {total_tokens} tokens ({tokens_per_sec:.2f} tokens/sec)")

if __name__ == "__main__":
    main()
