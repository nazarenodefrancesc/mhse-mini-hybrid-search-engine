import logging
import os
from typing import List

from hybrid_search_engine.model.document import Document
from hybrid_search_engine.retrievers import BM25Retriever, FaissRetriever
from hybrid_search_engine.rank_fusion import reciprocal_rank_fusion
from hybrid_search_engine.reranking import InHouseReranker, CohereReranker

log = logging.getLogger(__name__)

# https://gist.github.com/breadchris/b73aae81953eb8f865ebb4842a1c15b5#file-hybrid-py
# OPPURE: https://python.langchain.com/docs/how_to/hybrid/


class HybridSearch:
    def __init__(self, documents: list, hybrid_search_active: bool = False, language: str = None, reranker: str = "inhouse", embedding_model: str = "openai"):
        self.hybrid_search_active = hybrid_search_active

        if len(documents) > 0 and isinstance(documents[0], str):
            log.info("Converting list of strings to list of Documents. Id will be hash(content), no title, no metadata.")
            documents = [Document(content=doc) for doc in documents]

        self.documents: List[Document] = documents

        log.info(f"hybrid_search_active: {hybrid_search_active}")
        log.info(f"Number of documents: {len(documents)}")

        # Create the BM25 model and index the corpus
        self.bm25_retriever = BM25Retriever(documents, language=language)

        if hybrid_search_active:

            self.faiss_retriever = FaissRetriever(documents, embedding_model=embedding_model)

            # Reranker
            if reranker == "inhouse":
                log.info("Using InHouseReranker")
                self.reranker = InHouseReranker()
            elif reranker == "cohere":
                log.info("Using CohereReranker")
                self.reranker = CohereReranker(api_key=os.getenv("COHERE_API_KEY"))

    def add_documents(self, new_docs: list):

        if len(new_docs) > 0 and isinstance(new_docs[0], str):
            log.info("Converting list of strings to list of Documents. Id will be hash(content), no title, no metadata.")
            new_docs = [Document(content=doc) for doc in new_docs]

        log.info(f"Adding {len(new_docs)} documents to the index... Previous number of documents: {len(self.documents)}")

        self.documents += new_docs
        self.bm25_retriever.add_documents(new_docs)

        if self.hybrid_search_active:
            self.faiss_retriever.add_documents(new_docs)

        log.info(f"New number of documents: {len(self.documents)}")

    def search(self, query, rows: int = 10, top_k: int = 50, rank_fusion_k: int = 60):

        if not self.hybrid_search_active:
            top_k = rows

        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
        bm25results_ids, scores = self.bm25_retriever.retrieve(query, top_k=top_k)
        results_ids = bm25results_ids
        # TODO: restituiscono la posizione del documento nel corpus, non l'id. RESTITUIRE L'ID (hash del contenuto)

        if self.hybrid_search_active:

            faiss_results_ids = self.faiss_retriever.retrieve(query, top_k=top_k)

            # Rank Fusion
            fused_results_with_scores, doc_ids = reciprocal_rank_fusion(bm25results_ids, faiss_results_ids, k=rank_fusion_k)
            results = self.get_documents_from_ids(doc_ids)
            scores = [f["score"] for f in fused_results_with_scores]

            # Reranking, ma solo dei rows migliori
            reranked_result_idcs = self.reranker.rerank(query, results[:rows])
            results_ids = [r.doc_id for r in reranked_result_idcs]

        # TODO:
        # - query in parallelo con concurrent.futures

        return self.get_documents_from_ids(results_ids)[:rows], scores[:rows]

    def get_documents_from_ids(self, doc_ids):
        list_docs = []
        for doc_id in doc_ids:
            for doc in self.documents:
                if doc.id == doc_id:
                    list_docs.append(doc)
                    break
        return list_docs




