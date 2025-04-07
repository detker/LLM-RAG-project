from typing import Tuple, List, Dict, Any, Union

import numpy as np
import pandas as pd
from numpy import floating
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.DataWrangler import DataWrangler
from src.fixed_token_chunker import FixedTokenChunker
from src.recursive_token_chunker import RecursiveTokenChunker
from src.Embedding import SentenceTransformersEmbedding, GPTEmbedding
from src.RetrievalEvaluator import RetrievalEvaluator
from src.RerankerCrossEncoder import RerankerCrossEncoder
from src.Constants import RERANKER_MULTIPLIER, MAX_WORKERS


class EvaluationPipeline:
    """
    End-to-end pipeline for evaluating information retrieval performance.

    This class provides a comprehensive workflow for evaluating document retrieval systems:
    1. Chunk documents into smaller segments
    2. Generate embeddings for documents and queries
    3. Calculate similarity scores between queries and documents
    4. Retrieve top-k most similar documents
    5. Optionally rerank documents using a cross-encoder model
    6. Evaluate retrieval quality using standard metrics (IoU, recall, precision)

    The pipeline supports multi-threaded document reranking for improved performance.
    """

    def __init__(self,
                 chunker: Union[FixedTokenChunker, RecursiveTokenChunker],
                 embd_func: Union[SentenceTransformersEmbedding, GPTEmbedding],
                 top_k: int = 3,
                 reranker: bool = False) -> None:
        """
        Initialize the evaluation pipeline with specified components.

        :param chunker: Document chunking instance
        :type chunker: Union[FixedTokenChunker, RecursiveTokenChunker]
        :param embd_func: Embedder instance for generating embeddings from text
        :type embd_func: Union[SentenceTransformersEmbedding, GPTEmbedding]
        :param top_k: Number of top documents to retrieve for each query (default: 3)
        :type top_k: int
        :param reranker: Whether to use cross-encoder reranking (default: False)
        :type reranker: bool
        """
        self.chunker = chunker
        self.embd_func = embd_func
        self.top_k = top_k

        self.data_wrangler = DataWrangler()
        self.retrieval_evaluator = RetrievalEvaluator()
        self.reranker = RerankerCrossEncoder() if reranker else None

    def chunk_corpora(self,
                      corpus_id: str) -> Tuple[List[str], List[Dict[str, int]], pd.DataFrame]:
        """
        Load and chunk the corpus identified by corpus_id.

        :param corpus_id: Identifier of the corpus to process
        :type corpus_id: str
        :return: Tuple containing (list of chunks, list of chunk metadata, DataFrame of questions)
        :rtype: Tuple[List[str], List[Dict[str, int]], pd.DataFrame]
        :raises RuntimeError: If chunking fails
        """
        try:
            self.data_wrangler.load_corpora_and_questions(corpus_id)
            return self.data_wrangler.construct_chunks_and_meta(self.chunker)
        except Exception as e:
            raise RuntimeError(f"Failed to perform chunking: {e}")

    def get_embeddings(self,
                       docs: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text documents.

        :param docs: List of text documents/chunks to embed
        :type docs: List[str]
        :return: Array of document embeddings
        :rtype: np.ndarray
        """
        return self.embd_func(docs, normalize=True)

    def calculate_similarities(self,
                               queries_embds: np.ndarray,
                               doc_embds: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity scores between query and document embeddings.

        :param queries_embds: Array of query embeddings
        :type queries_embds: np.ndarray
        :param doc_embds: Array of document embeddings
        :type doc_embds: np.ndarray
        :return: Similarity matrix where similarity[i,j] is the similarity between
                 document i and query j
        :rtype: np.ndarray
        :raises RuntimeError: If similarity calculation fails
        """
        try:
            return np.dot(doc_embds, queries_embds.T)
        except Exception as e:
            raise RuntimeError(f"Failed to calculate similarities: {e}")

    def retrieve_top_k(self,
                       similarities: np.ndarray) -> List[List[int]]:
        """
        Retrieve indices of the top-k most similar documents for each query.

        :param similarities: Similarity matrix from calculate_similarities
        :type similarities: np.ndarray
        :return: List of lists, where each inner list contains top-k document indices
                 for a single query
        :rtype: List[List[int]]
        :raises ValueError: If top_k is invalid
        :raises RuntimeError: If retrieval fails
        """
        if self.top_k <= 0 or self.top_k > similarities.shape[0]:
            raise ValueError(f"top_k must be between 1 and {len(similarities)}")
        try:
            return np.argsort(similarities, axis=0)[-self.top_k:][::-1].T.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve top_k: {e}")

    def _rerank_single_query(self,
                             args) -> List[Dict[str, Any]]:
        """
        Rerank documents for a single query using the cross-encoder.

        Helper method used for parallel processing of reranking tasks.

        :param args: Tuple containing (query_index, query_text, list_of_documents)
        :type args: tuple
        :return: List of reranked documents
        :rtype: List[Dict[str, Any]]
        """
        _, query, docs = args
        return self.reranker(query, docs, self.top_k // RERANKER_MULTIPLIER)

    def evaluate_retrievals(self,
                            corpus_id: str,
                            min_top_k: bool = False) -> dict[str, floating[Any]]:
        """
        Evaluate retrieval performance for a corpus using the complete pipeline.

        This method executes the full retrieval and evaluation workflow:
        1. Chunk the corpus
        2. Generate embeddings for chunks and queries
        3. Calculate similarities and retrieve top-k chunks
        4. Optionally rerank chunks using the cross-encoder
        5. Evaluate retrieval quality using standard metrics

        :param corpus_id: Identifier of the corpus to evaluate
        :type corpus_id: str
        :param min_top_k: Whether to use minimum number of top chunks for evaluation
                         (default: False)
        :type min_top_k: bool
        :return: Dictionary of mean evaluation metrics (iou_score, recall_score, precision_score)
        :rtype: Dict[str, float]
        :raises SystemExit: If evaluation fails
        """
        try:
            if self.reranker:
                self.top_k = self.top_k * RERANKER_MULTIPLIER
            chunks, chunk_metadatas, queries_df = self.chunk_corpora(corpus_id)
            queries_list = queries_df['question'].tolist()
            docs_embds = self.get_embeddings(chunks)
            queries_embds = self.get_embeddings(queries_list)
            similarities = self.calculate_similarities(queries_embds, docs_embds)
            top_k_indices = self.retrieve_top_k(similarities)

            top_chunks = [[{'chunk': chunks[i],
                            'chunk_meta': chunk_metadatas[i],
                            'similarity': similarities[i, j]}
                           for i in top_k_indices[j]] for j in range(0, len(top_k_indices))]

            if self.reranker:
                print("Reranking documents...")
                with tqdm(total=len(queries_list), desc="Reranking Progress") as pbar:
                    rerank_args = [(i, queries_list[i], top_chunks[i])
                                   for i in range(len(queries_list))]
                    reranked_results = []

                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

                        future_to_query = {
                            executor.submit(self._rerank_single_query, args): args
                            for args in rerank_args
                        }

                        for future in as_completed(future_to_query):
                            args = future_to_query[future]
                            idx = args[0]
                            try:
                                result = future.result()
                                reranked_results.append((idx, result))
                            except Exception as e:
                                print(f'Error processing query {idx}: {str(e)}')
                            pbar.update(1)

                reranked_results.sort(key=lambda x: x[0])
                top_chunks = [result[1] for result in reranked_results]

            top_chunks = np.array(top_chunks)

            metrics = self.retrieval_evaluator.evaluate_retrievals(top_chunks, queries_df, min_top_k)
            metrics_mean = {k: np.mean(v) for k, v in metrics.items()}

            return metrics_mean

        except Exception as ex:
            print(f"Error during evaluation: {ex}")
            exit(1)
