from typing import List

import bm25s
import logging

import faiss
from numpy import array
from Stemmer import Stemmer
from hybrid_search_engine.language import LanguageDetector
from hybrid_search_engine.model.document import Document
from hybrid_search_engine.embeddings import SentenceTransformerEmbedder, OpenAIEmbedder


log = logging.getLogger(__name__)



class BaseRetriever:

        def __init__(self, documents: List[Document]):
            self.documents = documents.copy()

        def _get_text_corpus(self):
            return [doc.get_searchable_text() for doc in self.documents]

        def add_documents(self, new_docs: List[Document]):
            """
            Add documents to the retriever, updating the index
            :param new_docs:
            :return:
            """
            raise NotImplementedError

        def retrieve(self, query, top_k=10):
            """
            Retrieve documents based on a query
            :param query:
            :param top_k:
            :return:
                list of document indices
            """
            raise NotImplementedError


class BM25Retriever(BaseRetriever):

    def __init__(self, documents, language: str = None):
        super().__init__(documents)

        self.language = language
        if not self.language:
            log.info(f"Language not provided, detecting language of the corpus...")
            self.language = self._detect_language(documents)
            log.info(f"Detected language: {self.language}")

        # https://github.com/xhluca/bm25s
        corpus_tokens = bm25s.tokenize(self._get_text_corpus(), stopwords=language, stemmer=Stemmer(self.language))
        # corpus_tokens = []
        # for doc in documents:
        #     doc_language = self._detect_language([doc])
        #     corpus_tokens.append(bm25s.tokenize(doc, stopwords=doc_language, stemmer=Stemmer(doc_language)))

        self.bm25_retriever = bm25s.BM25()
        self.bm25_retriever.index(corpus_tokens)

    @staticmethod
    def _detect_language(docs):
        concatenated_docs = "\n".join([doc.get_searchable_text() for doc in docs])
        language_detector = LanguageDetector().detect_language_of(concatenated_docs)
        return language_detector.lower()

    def add_documents(self, new_docs: List[Document]):
        self.documents += new_docs
        new_docs_language = self._detect_language(new_docs)
        log.info(f"New docs language: {new_docs_language}")

        corpus_tokens = bm25s.tokenize(self._get_text_corpus(), stopwords=new_docs_language, stemmer=Stemmer(new_docs_language))
        self.bm25_retriever = bm25s.BM25()
        self.bm25_retriever.index(corpus_tokens)

    def retrieve(self, query, top_k=10):
        query_language = self._detect_language([Document(content=query)])
        query_tokens = bm25s.tokenize(query, stemmer=Stemmer(query_language), stopwords=query_language)

        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
        # bm25results, scores = self.bm25_retriever.retrieve(query_tokens, corpus=self._get_text_corpus(), k=top_k, n_threads=-1) # num_thread=-1 to use all available threads
        bm25results_indexes, scores = self.bm25_retriever.retrieve(query_tokens, k=top_k, n_threads=-1) # num_thread=-1 to use all available threads
        bm25results_indexes = [r for r in bm25results_indexes[0]]
        bm25results_ids = [self.documents[i].id for i in bm25results_indexes]

        return bm25results_ids, scores


class FaissRetriever(BaseRetriever):

    def __init__(self, documents, embedding_model: str = "openai"):
        super().__init__(documents)

        logging.info(f"Embedding model: {embedding_model}")
        # Sentence transformer for embeddings
        self.embedder = SentenceTransformerEmbedder() if embedding_model == "sentence-transformers" else OpenAIEmbedder()
        document_embeddings = self.embedder.embed(self._get_text_corpus())

        # FAISS initialization
        self.faiss_index = faiss.IndexFlatL2(document_embeddings.shape[1])
        self.faiss_index.add(array(document_embeddings).astype('float32'))

    def add_documents(self, new_docs: List[Document]):
        self.documents += new_docs
        new_text_corpus = [doc.content for doc in new_docs]
        new_doc_embeddings = self.embedder.embed(new_text_corpus)
        self.faiss_index.add(array(new_doc_embeddings).astype('float32'))

    def retrieve(self, query, top_k=10):
        query_embedding = self.embedder.embed([query])

        # FAISS search on the top documents
        _, ranked_indices = self.faiss_index.search(array(query_embedding).astype('float32'), top_k)
        ranked_indices = ranked_indices[0]
        ranked_ids = [self.documents[i].id for i in ranked_indices]

        return ranked_ids
