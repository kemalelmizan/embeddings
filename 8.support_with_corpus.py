import numpy as np
import ollama
import time

def embed(texts, model="nomic-embed-text"):
    res = ollama.embed(model=model, input=texts)
    return np.array(res["embeddings"])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_best_passage(query, corpus):
    vecs = embed([query] + corpus)
    query_vec = vecs[0]
    corpus_vecs = vecs[1:]
    scores = [cosine_similarity(query_vec, vec) for vec in corpus_vecs]
    best_idx = int(np.argmax(scores))
    return corpus[best_idx], scores[best_idx]

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

    start_time = time.time()
    best_passage, score = retrieve_best_passage(query, corpus)
    response = generate_response(query, best_passage)
    elapsed = time.time() - start_time
    total_tokens = sum(len(c.split()) for c in corpus) + len(query.split())
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else float('inf')

    print(f"User Query: {query}")
    print(f"Matched Passage (score {score:.4f}): {best_passage}")
    print("\nAssistant Response:")
    print(response)
    print(f"\nTime taken: {elapsed:.4f} seconds for {total_tokens} tokens ({tokens_per_sec:.2f} tokens/sec)")

if __name__ == "__main__":
    main()
