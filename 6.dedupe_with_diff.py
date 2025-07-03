import numpy as np
import ollama
from itertools import combinations
import time

def embed(texts, model="nomic-embed-text"):
    res = ollama.embed(model=model, input=texts)
    return np.array(res["embeddings"])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_near_duplicates(texts, threshold=0.9):
    embeddings = embed(texts)
    pairs = []
    for i, j in combinations(range(len(texts)), 2):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        if sim > threshold:
            pairs.append((i, j, sim))
    return pairs

def explain_difference(text_a, text_b, model="gemma3"):
    prompt = f"""
These two texts are very similar. Highlight their differences.

Text A: "{text_a}"
Text B: "{text_b}"

Differences:
"""
    res = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return res["message"]["content"].strip()

def main():
    texts = [
        "The iPhone 15 Pro features a titanium frame and improved GPU.",
        "Apple's iPhone 15 Pro has a titanium build and faster graphics chip.",
        "The Samsung Galaxy S24 introduces advanced AI photo editing features.",
        "Samsung's Galaxy S24 comes with AI-powered camera tools for editing.",
        "The iPhone 14 Pro was released with a stainless steel frame.",
    ]

    print("\U0001F50D Finding near-duplicates...\n")
    start_time = time.time()
    pairs = find_near_duplicates(texts)
    elapsed = time.time() - start_time
    total_tokens = sum(len(t.split()) for t in texts)
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else float('inf')

    if not pairs:
        print("No near-duplicates found.")
        print(f"\nTime taken: {elapsed:.4f} seconds for {total_tokens} tokens ({tokens_per_sec:.2f} tokens/sec)")
        return

    for i, j, score in pairs:
        print(f"\U0001F9E0 Match: {score:.4f}")
        print(f"Text A: {texts[i]}")
        print(f"Text B: {texts[j]}")
        explanation = explain_difference(texts[i], texts[j])
        print("💬 Differences:\n", explanation)
        print("-" * 80)
    print(f"\nTime taken: {elapsed:.4f} seconds for {total_tokens} tokens ({tokens_per_sec:.2f} tokens/sec)")

if __name__ == "__main__":
    main()
