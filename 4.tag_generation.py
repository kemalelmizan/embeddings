import ollama
import numpy as np

def embed(texts, model="nomic-embed-text"):
    res = ollama.embed(model=model, input=texts)
    return np.array(res["embeddings"])

def generate_tags(texts, model="gemma3"):
    joined = "\n".join(f"- {t}" for t in texts)
    prompt = f"""
Generate 3–5 descriptive tags for the following texts. Focus on topics, domains, or genres:

{joined}

Tags:"""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

def main():
    documents = [
        "Dune is a science fiction novel by Frank Herbert set in a desert world.",
        "The Hobbit is a fantasy story by J.R.R. Tolkien featuring a journey and a dragon.",
        "Neuromancer is a cyberpunk novel by William Gibson involving hackers and AI.",
        "Pride and Prejudice is a romantic novel by Jane Austen about societal expectations.",
        "Emma is another romance by Jane Austen exploring matchmaking in English society.",
    ]

    # Optionally: group similar docs first (e.g. by cosine similarity or clustering)
    print("🔍 Generating tags for entire collection:\n")
    tags = generate_tags(documents)
    print("Generated Tags:\n", tags)

if __name__ == "__main__":
    main()
