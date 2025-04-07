from typing import List, Any, Dict

import tiktoken
from openai import OpenAI

from src.Constants import OPENAI_API_KEY


class RerankerCrossEncoder:
    """
    A document reranker that uses LLMs as cross-encoders to score document relevance.

    This class leverages OpenAI's models to evaluate the relevance of documents with respect
    to a query, providing a more semantically-aware ranking than traditional methods.
    """

    def __init__(self,
                 model_name: str = 'gpt-4o-mini',
                 max_tokens: int = 2048) -> None:
        """
        Initialize the reranker with specified OpenAI model and token limit.

        :param model_name: Name of the OpenAI model to use (default: 'gpt-4o-mini')
        :type model_name: str
        :param max_tokens: Maximum number of tokens allowed in a prompt (default: 2048)
        :type max_tokens: int
        :raises ValueError: If OpenAI API key is not set
        :raises RuntimeError: If initialization fails
        """
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set")

        try:
            self.model_name = model_name
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.max_tokens = max_tokens
            self.tokenizer = tiktoken.get_encoding('cl100k_base')
        except Exception as ex:
            raise RuntimeError(f"Failed to initialize RerankerCrossEncoder: {ex}")

    def __call__(self,
                 query: str,
                 docs: List[Dict[str, Any]],
                 top_k: int) -> List[Dict[str, Any]]:
        """
        Rerank documents based on their relevance to the query.

        This method sends each document with the query to the LLM, which scores
        the document's relevance. Documents are then sorted by score and the top_k
        most relevant documents are returned.

        :param query: The search query
        :type query: str
        :param docs: List of document dictionaries, each containing at least a 'chunk' key
                    with the document text
        :type docs: List[Dict[str, Any]]
        :param top_k: Number of top documents to return after reranking
        :type top_k: int
        :return: List of the top_k most relevant document dictionaries
        :rtype: List[Dict[str, Any]]
        :raises ValueError: If query or docs is empty, or if top_k is invalid
        :raises RuntimeError: If the reranking process fails
        """
        if not query or not docs:
            raise ValueError("Query and documents list cannot be empty")
        if top_k <= 0 or top_k > len(docs):
            raise ValueError(f"Invalid top_k value: {top_k}")

        reranked_docs = []
        try:
            for idx, docdata in enumerate(docs):
                doc = docdata['chunk']
                prompt = self._construct_prompt(query, doc)
                prompt_tokens = self.tokenizer.encode(prompt)

                if len(prompt_tokens) > self.max_tokens:
                    print(f"Document exceeds max tokens: {len(prompt_tokens)} > {self.max_tokens}")
                    continue

                try:
                    res = self.client.chat.completions.create(
                        model=self.model_name,
                        temperature=0,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    score = float(res.choices[0].message.content.strip())
                    if not 0 <= score <= 1:
                        raise ValueError(f'Score out of range [0,1]: {score}')
                    reranked_docs.append((idx, doc, score))
                except Exception as ex:
                    print(f"Error processing document {doc}: {ex}")
                    continue

            reranked_docs = sorted(reranked_docs, key=lambda x: x[-1], reverse=True)[:top_k]
            return [docs[x[0]] for x in reranked_docs]
        except Exception as ex:
            raise RuntimeError(f"Failed to rerank documents: {ex}")

    def _construct_prompt(self,
                          query: str,
                          doc: str) -> str:
        return f"""Query: {query}\n
                   Document: {doc}\n
                   Rate the relevance of the document to the query on a scale from 0 to 1, 
                   where 0 is not relevant and 1 is highly relevant. Only output a single 
                   floating-point number as the answer."""
