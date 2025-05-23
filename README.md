# Embeddings
Various embedding examples with ollama, uv, gemma3 and nomic-embed-text

## Setup

1. Install [ollama](https://ollama.com/download), then:
```
ollama pull gemma3 nomic-text-embed
```

2. Install [uv](https://docs.astral.sh/uv/#installation), then
```
uv venv
source .venv/bin/activate  
uv init
uv add ollama numpy scikit-learn

uv run 1.similarity.py
```