import logging
import os
from time import sleep
from typing import List

from cohere import Client
from transformers import AutoModelForSequenceClassification

from hybrid_search_engine.model.document import Document


class DocWithScore:
    def __init__(self, doc_id: int, score: float):
        self.doc_id = doc_id
        self.score = score

    def __str__(self):
        return f"{self.doc_id} - {self.score}"

class Reranker:

    def rerank(self, query: str, documents: List[Document]):
        raise NotImplementedError


class InHouseReranker(Reranker):

        def __init__(self, model_name: str = "jinaai/jina-reranker-v2-base-multilingual", device: str = 'cpu'):
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype="auto",
                trust_remote_code=True,
            )
            self.model.to(device)  # or 'cpu' if no GPU is available
            self.model.eval()

        def rerank(self, query: str, documents: List[Document]):
            sentence_pairs = [[query, doc.get_searchable_text()] for doc in documents]
            scores = self.model.compute_score(sentence_pairs, max_length=1024)
            sorted_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            doc_with_scores = [DocWithScore(doc.id, score) for doc, score in sorted_results]
            return doc_with_scores

class CohereReranker(Reranker):

    def __init__(self, api_key: str, model_name: str = "rerank-v3.5"):
        self.client = Client(api_key)
        self.alternative_client = Client(os.getenv("COHERE_ALTERNATIVE_API_KEY")) if os.getenv("COHERE_ALTERNATIVE_API_KEY") else None
        self.model_name = model_name

    def rerank(self, query: str, documents: List[Document]):

        document_contents = [doc.get_searchable_text() for doc in documents]

        error = True
        retry = 3
        result_ids = None

        # TODO: codice di cacca, rifare
        while error and retry > 0:
            try:
                # la free trial di cohere permette 10 chiamate al minuto
                try:
                    result_ids = self.client.rerank(model=self.model_name, query=query, documents=document_contents, return_documents=False)
                    break
                except Exception as e:
                    logging.error(f"Error: {e}")
                    if self.alternative_client:
                        logging.info("Error occurred, using alternative API key")
                        result_ids = self.alternative_client.rerank(model=self.model_name, query=query, documents=document_contents, return_documents=False)
                        break

            except Exception as e:
                logging.error(f"Error: {e}")
                # rate limit: https://docs.cohere.com/docs/rate-limits?_gl=1*1glhfnq*_ga*NTM1NzQ3MDkxLjE3Mjk4NjI1Nzk.*_ga_CRGS116RZS*MTcyOTg2MjU3Ny4xLjEuMTcyOTg2NDc1NS41Ni4wLjA.*_gcl_au*Nzc5MjczMjMyLjE3Mjk4NjI1OTA.
                sleep(10)
                retry -= 1

        if not result_ids:
            print(f"ERROR: returning documents without scores")
            return [DocWithScore(d.id, 0) for d in documents]

        doc_with_scores = [DocWithScore(documents[r.index].id, r.relevance_score) for r in result_ids.results]
        return doc_with_scores


if __name__ == "__main__":

    # Example query and documents
    query = "Cosa sono i gatti?"
    documents = [
        Document(id=1, content="Organic skincare for sensitive skin with aloe vera and chamomile."),
        Document(id=2, content="I gatti sono animali domestici molto affettuosi."),
        Document(id=3, content="New makeup trends focus on bold colors and innovative techniques"),
        Document(id=4, content="Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille"),
        Document(id=5, content="Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken"),
        Document(id=6, content="Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla"),
        Document(id=7, content="Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras"),
        Document(id=8, content="针对敏感肌专门设计的天然有机护肤产品"),
        Document(id=9, content="新的化妆趋势注重鲜艳的颜色和创新的技巧"),
        Document(id=10, content="敏感肌のために特別に設計された天然有機スキンケア製品"),
        Document(id=11, content="新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています"),
    ]

    # Rerank documents
    reranker = InHouseReranker()
    sorted_results = reranker.rerank(query, documents)

    print([f"{idx}) {doc}" for idx, doc in enumerate(sorted_results)])

    # Cohere Reranker
    api_key = "wQlC4qU8qY7uIK7tcBW2ttUh7jJ8d2SXXtwiuxTB"
    reranker = CohereReranker(api_key)
    sorted_results = reranker.rerank(query, documents)

    print([f"{idx}) {doc}" for idx, doc in enumerate(sorted_results)])

