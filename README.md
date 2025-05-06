# Hybrid Search Engine

Hybrid Search Engine is a Python library and CLI application for performing hybrid searches by combining sparse retrieval (BM25) and dense retrieval (FAISS) with reranking capabilities (Cohere or in-house).

## Features

- Sparse retrieval using the `bm25s` library.
- Dense retrieval via FAISS and embedding models (OpenAI or Sentence Transformers).
- Rank fusion with Reciprocal Rank Fusion (RRF).
- Optional reranking with Cohere or an in-house implementation.
- Incremental document indexing.
- Automatic language detection and dedicated stemming.
- Modular, easily extensible architecture.

## Requirements

- Python 3.8 or higher.
- (Optional) API keys for OpenAI and Cohere if using external embedding or reranking services.

## Installation

```bash
git clone <repository_url>
cd <repository_directory>
pip install -r requirements.txt
pip install -e .
cp env.example .env
```

Populate the `.env` file with your API keys and desired configuration variables.

## Configuration

The main environment variables (defined in `.env`) are:

- `OPENAI_API_KEY`: API key for OpenAI embedding models.
- `COHERE_API_KEY`: API key for Cohere reranking.
- `RERANKER`: `cohere` or `inhouse` (default: `cohere`).
- `EMBEDDING`: `openai` or `sentence-transformers` (default: `openai`).
- `OPENAI_EMBEDDING_MODEL`: OpenAI embedding model (default: `text-embedding-3-small`).

## Usage Example

```python
from hybrid_search_engine.model.document import Document
from hybrid_search_engine.searcher import HybridSearch

# Define documents
docs = [
    Document(id="1", content="Content of document 1", metadata={"topic": "AI"}),
    Document(id="2", content="Content of document 2")
]

# Initialize the hybrid search engine
hs = HybridSearch(
    documents=docs,
    hybrid_search_active=True,
    reranker="cohere",
    embedding_model="openai"
)

# Execute a search
results, scores = hs.search("artificial intelligence", rows=5, top_k=50)
for doc, score in zip(results, scores):
    print(f"{doc.id}: {doc.content} (score: {score})")
```

## Project Structure

- `hybrid_search_engine/`: Library source code
  - `retrievers.py`: BM25 and FAISS retrieval modules
  - `rank_fusion.py`: Rank fusion functions
  - `reranking.py`: External or in-house reranking modules
  - `chunking.py`: Document chunking utilities
  - `embeddings.py`: Embedding model wrappers
  - `language.py`: Language detection and stemming
  - `searcher.py`: Main `HybridSearch` class
  - `model/document.py`: Document model definition
- `main.py`: Example script to test the search engine
- `test_data/`: Sample data for quick checks
- `env.example`: Template for environment variables

## Contributing

PRs and issue reports are welcome!

## License

This project is released under the Apache 2.0 License.