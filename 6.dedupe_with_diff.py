import numpy as np
import ollama
from itertools import combinations

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

    print("🔍 Finding near-duplicates...\n")
    pairs = find_near_duplicates(texts)

    if not pairs:
        print("No near-duplicates found.")
        return

    for i, j, score in pairs:
        print(f"🧠 Match: {score:.4f}")
        print(f"Text A: {texts[i]}")
        print(f"Text B: {texts[j]}")
        explanation = explain_difference(texts[i], texts[j])
        print("💬 Differences:\n", explanation)
        print("-" * 80)

if __name__ == "__main__":
    main()
