import numpy as np
from sklearn.cluster import KMeans
import ollama
import time

# Step 1: Embed texts using nomic-embed-text
def embed(texts, model="nomic-embed-text"):
    res = ollama.embed(model=model, input=texts)
    return np.array(res["embeddings"])

# Step 2: Cluster documents
def cluster_texts(texts, n_clusters=2):
    vectors = embed(texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(vectors)

    clusters = {i: [] for i in range(n_clusters)}
    for text, label in zip(texts, labels):
        clusters[label].append(text)
    return clusters

# Step 3: Summarize each cluster using Gemma
def summarize(texts, model="gemma3"):
    prompt = "Summarize the main topic of these texts:\n\n" + "\n".join(f"- {t}" for t in texts)
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip()

def main():
    documents = [
        "Dune is a science fiction novel by Frank Herbert.",
        "The Hobbit is a fantasy novel by J.R.R. Tolkien.",
        "Neuromancer is a cyberpunk novel by William Gibson.",
        "Pride and Prejudice is a romantic novel by Jane Austen.",
        "Foundation is a space opera series by Isaac Asimov.",
        "Emma is another romantic novel by Jane Austen.",
        "The Lord of the Rings is a high fantasy trilogy by Tolkien.",
    ]

    start_time = time.time()
    clusters = cluster_texts(documents, n_clusters=3)
    elapsed = time.time() - start_time
    total_tokens = sum(len(d.split()) for d in documents)
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else float('inf')

    for i, group in clusters.items():
        print(f"\n\U0001F9E0 Cluster {i+1} ({len(group)} items):")
        for doc in group:
            print(f"  - {doc}")
        summary = summarize(group)
        print(f"\U0001F4DD Summary: {summary}")
    print(f"\nTime taken: {elapsed:.4f} seconds for {total_tokens} tokens ({tokens_per_sec:.2f} tokens/sec)")

if __name__ == "__main__":
    main()
