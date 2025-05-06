import logging
import os

from dotenv import load_dotenv
from hybrid_search_engine.model.document import Document
from hybrid_search_engine.searcher import HybridSearch

logging.basicConfig(level=logging.INFO)

load_dotenv()


if __name__ == "__main__":

    reranker = os.getenv("RERANKER", "cohere")
    embedding = os.getenv("EMBEDDING", "openai")


    # Sample usage
    documents = [
        Document(id="01", content="L'intelligenza artificiale (AI) è la simulazione dell'intelligenza umana da parte di macchine.", metadata={"author": "John Doe"}),
        Document(id="02", content="Il riconoscimento ottico dei caratteri (OCR) è un processo che converte i documenti cartacei in documenti digitali."),
        Document(id="03", content="Il deep learning è una branca dell'apprendimento automatico basata su reti neurali artificiali."),
        Document(id="04", content="Il computer vision è una disciplina che si occupa di rendere i computer capaci di interpretare e comprendere il mondo visivo."),
        Document(id="05", content="Il Natural Language Processing (NLP) è una disciplina che si occupa di far comprendere ai computer il linguaggio umano in modo naturale."),
        Document(id="06", content="Il reinforcement learning è un tipo di apprendimento automatico che si basa su un sistema di ricompense e punizioni."),
        Document(id="07", content="Il transfer learning è una tecnica di apprendimento automatico in cui un modello addestrato su una certa attività viene riutilizzato per un'altra attività correlata.")
    ]

    logging.info("LOADING DOCUMENTS AND INDEXING...")
    # hybrid_search = HybridSearch(documents)

    # hs = HybridSearch(documents, hybrid_search_active=False)
    hs = HybridSearch(documents, hybrid_search_active=True, reranker=reranker, embedding_model=embedding)
    # hs = HybridSearch(documents, hybrid_search_active=True, reranker=reranker, embedding_model=embedding)

    queries = [
        "IA",
        "modelli visuali",
        "apprendimento per rinforzo"
    ]

    target_doc_ids = [
        "01",
        "04",
        "06"
    ]

    correct = 0
    mean_position = 0
    for query, target in zip(queries, target_doc_ids):
        logging.info(f"Query: {query}")
        results, scores = hs.search(query, rows=5, top_k=5)

        for result in results:
            if target == result.id:
                logging.info(f"Query '{query}' has a match with the target document '{target}' in position {results.index(result) + 1}")
                correct += 1
                mean_position += results.index(result) + 1
                break

    logging.info(f"Accuracy: {correct / len(queries)}")
    logging.info(f"Mean position: {mean_position / correct}")

    assert correct / len(queries) == 1.0
    assert mean_position / correct == 1.0

    # test indicizzazione incrementale
    hs.add_documents([Document(id="08", content="Life is what happens when you're busy making other plans.")])
    results, scores = hs.search("Life", rows=5, top_k=5)
    logging.info(results)
    logging.info(scores)

    assert results[0].id == "08"