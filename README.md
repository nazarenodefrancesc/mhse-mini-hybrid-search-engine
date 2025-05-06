# Hybrid Search Engine

Hybrid Search Engine è una libreria Python e applicazione CLI per eseguire ricerche ibride combinando retrieval sparse (BM25) e dense (FAISS) con funzionalità di reranking (Cohere o in-house).

## Caratteristiche
- Recupero BM25 tramite la libreria `bm25s`.
- Recupero denso tramite FAISS e modelli di embeddings (OpenAI o Sentence Transformers).
- Fusione dei ranking tramite Reciprocal Rank Fusion (RRF).
- Reranking opzionale con Cohere o implementazione interna.
- Indicizzazione incrementale dei documenti.
- Rilevamento automatico della lingua e stemming dedicato.
- Architettura modulare e facilmente estendibile.

## Requisiti
- Python 3.8 o superiore.
- Chiavi API (opzionali) per OpenAI e Cohere se si utilizzano embedding e reranking esterni.

## Installazione
```bash
git clone <repository_url>
cd <repository_directory>
pip install -r requirements.txt
pip install -e .
cp env.example .env
```
Compilare il file `.env` con le proprie chiavi API e configurare le variabili desiderate.

## Configurazione
Le principali variabili d'ambiente sono definite nel file `.env`:
- `OPENAI_API_KEY`: chiave API per i modelli di embedding OpenAI.
- `COHERE_API_KEY`: chiave API per il reranking con Cohere.
- `RERANKER`: `cohere` o `inhouse` (default: `cohere`).
- `EMBEDDING`: `openai` o `sentence-transformers` (default: `openai`).
- `OPENAI_EMBEDDING_MODEL`: modello embedding OpenAI (default: `text-embedding-3-small`).

## Esempio di utilizzo
```python
from hybrid_search_engine.model.document import Document
from hybrid_search_engine.searcher import HybridSearch

# Definizione dei documenti
docs = [
    Document(id="1", content="Contenuto del documento 1", metadata={"topic": "AI"}),
    Document(id="2", content="Contenuto del documento 2")
]

# Inizializzazione del motore di ricerca ibrido
hs = HybridSearch(
    documents=docs,
    hybrid_search_active=True,
    reranker="cohere",
    embedding_model="openai"
)

# Esecuzione della ricerca
results, scores = hs.search("intelligenza artificiale", rows=5, top_k=50)
for doc, score in zip(results, scores):
    print(f"{doc.id}: {doc.content} (score: {score})")
```

## Struttura del progetto
- `hybrid_search_engine/`: codice sorgente della libreria.
  - `retrievers.py`: moduli di retrieval BM25 e FAISS.
  - `rank_fusion.py`: funzioni per la fusione dei ranking.
  - `reranking.py`: moduli per il reranking esterno o interno.
  - `chunking.py`: funzioni di suddivisione in chunk.
  - `embeddings.py`: wrapper per modelli di embedding.
  - `language.py`: rilevamento lingua e stemming.
  - `searcher.py`: classe principale `HybridSearch`.
  - `model/document.py`: definizione del modello Documento.
- `main.py`: esempio di script per testare il motore di ricerca.
- `test_data/`: dati di test per verifiche rapide.
- `env.example`: template per le variabili d'ambiente.

## Contribuire
PR e segnalazioni di issue sono ben accetti.

## Licenza
Questo progetto è rilasciato sotto licenza Apache 2.0.
