from abc import abstractmethod
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from .Constants import EMBD_BATCH_SIZE, OPENAI_API_KEY


class BaseEmbedding:
    """
    Abstract base class for text embedding models.

    Defines the interface for embedding classes that convert text documents into
    numerical vector representations.
    """

    @abstractmethod
    def __init__(self,
                 model_name: str) -> None:
        if not model_name:
            raise ValueError("Model name cannot be empty.")
        self.model_name = model_name

    def __call__(self,
                 docs: List[str],
                 normalize: bool) -> Optional[np.ndarray]:
        return None


class SentenceTransformersEmbedding(BaseEmbedding):
    """
    Implementation of BaseEmbedding using the sentence-transformers library.

    This class generates embeddings using models from the sentence-transformers
    library, which provides state-of-the-art text embeddings.
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2') -> None:
        """
        Initialize the SentenceTransformers embedding model.

        :param model_name: Name of the sentence-transformers model to use (default: 'all-MiniLM-L6-v2')
        :type model_name: str
        :raises RuntimeError: If loading the model fails
        """
        super().__init__(model_name)
        try:
            self.model = SentenceTransformer(('sentence-transformers/' + model_name))
            self.embd_dims = self.model.get_sentence_embedding_dimension()
        except Exception as ex:
            raise RuntimeError(f"Failed to load SentenceTransformer model: {ex}")

    def __call__(self,
                 docs: List[str],
                 normalize: bool) -> np.ndarray:
        """
        Generate embeddings for the provided documents using sentence-transformers.

        :param docs: List of text documents to embed
        :type docs: List[str]
        :param normalize: Whether to normalize the resulting embeddings
        :type normalize: bool
        :return: Array of embeddings with shape [len(docs), embedding_dimension]
        :rtype: np.ndarray
        :raises RuntimeError: If embedding generation fails
        """
        try:
            return self.model.encode(docs, batch_size=EMBD_BATCH_SIZE, normalize_embeddings=normalize)
        except Exception as ex:
            raise RuntimeError(f"Failed to generate embeddings: {ex}")


class GPTEmbedding(BaseEmbedding):
    """
    Implementation of BaseEmbedding using OpenAI's embedding models.

    This class generates embeddings using OpenAI's API, which provides
    high-quality text representations for various NLP tasks.
    """

    def __init__(self,
                 model_name: str = 'text-embedding-3-large') -> None:
        """
        Initialize the GPT embedding model.

        :param model_name: OpenAI embedding model name to use (default: 'text-embedding-3-large')
        :type model_name: str
        :raises ValueError: If OpenAI API key is not set
        :raises RuntimeError: If initializing the OpenAI client fails
        """
        super().__init__(model_name)
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set.")
        try:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception as ex:
            raise RuntimeError(f"Failed to initialize OpenAI client: {ex}")

    def __call__(self,
                 docs: List[str],
                 normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for the provided documents using OpenAI's API.

        :param docs: List of text documents to embed
        :type docs: List[str]
        :param normalize: Ignored parameter whether to normalize the resulting embeddings as GPT embeddings are normalized by default
        :type normalize: bool
        :return: Array of embeddings with shape [len(docs), embedding_dimension]
        :rtype: np.ndarray
        :raises RuntimeError: If the API call fails
        """
        try:
            res = self.client.embeddings.create(input=docs, model=self.model_name)
            return np.array([r.embedding for r in res.data])
        except Exception as ex:
            raise RuntimeError(f"Failed to generate embeddings using OpenAI API: {ex}")
