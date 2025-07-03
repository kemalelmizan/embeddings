# Embeddings
Various embedding examples with ollama, uv, gemma3 and nomic-embed-text.
Mostly for Embedding + Retrieval flow: Query → Embed → Retrieve Best → Pass to LLM for natural response.

## Setup

1. Install [ollama](https://ollama.com/download), then:
```
ollama pull gemma3
ollama pull nomic-text-embed
```

2. Install [uv](https://docs.astral.sh/uv/#installation), then
```
uv venv
source .venv/bin/activate
uv add ollama numpy scikit-learn faiss-cpu

uv run 1.similarity.py

deactivate
```