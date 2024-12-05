import logging
import os
from typing import List

import numpy as np


class BaseEmbedder:

    dimensions = None

    def embed(self, texts: List[str]):
        """
        Embed a list of texts
        :param texts:
        :return:
            numpy array of shape (n_texts, dimensions)
        """
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):

    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = os.getenv("SENTECE_TRANSFORMER_MODEL") if os.getenv("SENTECE_TRANSFORMER_MODEL") else model_name
        self.embedder = SentenceTransformer(model_name)
        self.dimensions = self.embedder.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]):
        return self.embedder.encode(texts)

class OpenAIEmbedder(BaseEmbedder):

    def __init__(self, model_name: str = "text-embedding-3-small", provider: str = "openai", dimensions: int = None):
        from openai import Client, AzureOpenAI

        self.model_name = os.getenv("OPENAI_EMBEDDING_MODEL") if os.getenv("OPENAI_EMBEDDING_MODEL") else model_name

        if provider == "openai":
            self.embedder = Client()
        elif provider == "azure":
            self.embedder = AzureOpenAI(
                api_version=os.getenv("OPENAI_API_VERSION"),
                api_key=os.getenv("OPENAI_API_KEY"),
                azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT")
            )
        else:
            raise ValueError(f"Provider {provider} not supported")

        self.model_name = model_name
        if not dimensions:
            if model_name == "text-embedding-3-small":
                self.dimensions = 1536
            elif model_name == "text-embedding-3-large":
                self.dimensions = 3072
        else:
            logging.info(f"Using custom dimensions: {dimensions}")
            self.dimensions = dimensions
            # support custom dimensions https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-dimensions

    def embed(self, texts: List[str]):
        embeddings = [self.embedder.embeddings.create(input=text, model=self.model_name) for text in texts]
        embeddings =  [emb.data[0].embedding for emb in embeddings]
        embeddings = [np.array(emb) for emb in embeddings]
        embeddings = np.array(embeddings)

        return embeddings

