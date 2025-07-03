import numpy as np
import ollama
import time

def embed(texts, model="nomic-embed-text"):
    res = ollama.embed(model=model, input=texts)
    return np.array(res["embeddings"])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def match_users_to_profiles(users, profiles):
    user_vecs = embed(users)
    profile_vecs = embed(profiles)

    results = []
    for i, user_vec in enumerate(user_vecs):
        scores = [cosine_similarity(user_vec, prof_vec) for prof_vec in profile_vecs]
        best_idx = int(np.argmax(scores))
        results.append({
            "user_idx": i,
            "profile_idx": best_idx,
            "score": scores[best_idx]
        })
    return results

def explain_match(user_text, profile_text, model="gemma3"):
    prompt = f"""
Explain in 1 short sentence why this user profile is a good match for this opportunity.

User: "{user_text}"
Profile: "{profile_text}"

Explanation:"""
    res = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return res["message"]["content"].strip()

def main():
    users = [
        "Frontend developer with experience in React and Tailwind.",
        "AI enthusiast looking to work on machine learning research.",
        "Creative writer interested in fantasy and sci-fi storytelling.",
        "Full-stack engineer familiar with Node.js, Python, and databases."
    ]

    profiles = [
        "Join our startup building a React web app for e-commerce.",
        "We are hiring a data scientist for our AI lab.",
        "Looking for a novelist to write stories for a new RPG game.",
        "Seeking software engineer to manage backend and frontend APIs."
    ]

    start_time = time.time()
    results = match_users_to_profiles(users, profiles)
    elapsed = time.time() - start_time
    total_tokens = sum(len(u.split()) for u in users) + sum(len(p.split()) for p in profiles)
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else float('inf')

    for result in results:
        user_idx = result["user_idx"]
        profile_idx = result["profile_idx"]
        score = result["score"]
        print(f"User {user_idx+1} best matches Profile {profile_idx+1} (score {score:.4f})")
    print(f"\nTime taken: {elapsed:.4f} seconds for {total_tokens} tokens ({tokens_per_sec:.2f} tokens/sec)")

if __name__ == "__main__":
    main()
