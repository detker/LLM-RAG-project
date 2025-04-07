import argparse
import json
import os

from src.Constants import CHUNK_SIZE, CHUNK_OVERLAP, MINIMALIZE_CHUNKS
from src.fixed_token_chunker import FixedTokenChunker
from src.recursive_token_chunker import RecursiveTokenChunker
from src.Embedding import SentenceTransformersEmbedding, GPTEmbedding
from src.EvaluationPipeline import EvaluationPipeline
from src.Utils import Utils


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Configure command-line arguments for the evaluator.

    :param parser: ArgumentParser instance to configure
    :type parser: argparse.ArgumentParser
    """
    parser.add_argument('--corpus',
                        type=str,
                        default='wikitexts',
                        help='Corpus name (chatlogs|state_of_the_union|wikitexts)')
    parser.add_argument('--chunker',
                        type=str,
                        default='fixed_size',
                        help='Chunker strategy (fixed_size|recursive)')
    parser.add_argument('--embedding_model',
                        type=str,
                        default='all-MiniLM-L6-v2',
                        help="""Embedding model name (all-MiniLM-L6-v2|
                                                      multi-qa-mpnet-base-dot-v1|
                                                      text-embedding-3-small|
                                                      text-embedding-3-large)""")
    parser.add_argument('--top_k',
                        type=int,
                        default=5,
                        help='Top k retrievals')
    parser.add_argument('--reranker',
                        type=str,
                        default="no",
                        help='Enable reranker for retrieval (yes|no)')


def main():
    """
    Main function that runs the evaluation pipeline.

    The function:
    1. Parses command-line arguments
    2. Initializes the appropriate chunking strategy
    3. Sets up the embedding model
    4. Creates and runs the evaluation pipeline
    5. Outputs results to console and file

    :raises SystemExit: If initialization of components fails
    """
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    corpus_id: str = args.corpus
    chunker_name: str = args.chunker
    embd_model_name: str = args.embedding_model
    k: int = args.top_k
    reranker: bool = True if args.reranker == 'yes' else False

    try:
        if chunker_name == 'fixed_size':
            chunker = FixedTokenChunker(chunk_size=CHUNK_SIZE,
                                        chunk_overlap=CHUNK_OVERLAP,
                                        length_function=Utils.cl100k_base_length)
        else:
            chunker = RecursiveTokenChunker(chunk_size=CHUNK_SIZE,
                                            chunk_overlap=CHUNK_OVERLAP,
                                            length_function=Utils.cl100k_base_length)
    except Exception as ex:
        print(f"Error initializing chunker: {ex}")
        exit(1)

    try:
        if embd_model_name == 'text-embedding-3-large' or embd_model_name == 'text-embedding-3-small':
            embd_func = GPTEmbedding(embd_model_name)
        else:
            embd_func = SentenceTransformersEmbedding(embd_model_name)
    except Exception as ex:
        print(f"Error initializing embedding model: {ex}")
        exit(1)

    pipeline = EvaluationPipeline(chunker, embd_func, k, reranker)
    metrics = pipeline.evaluate_retrievals(corpus_id, MINIMALIZE_CHUNKS)

    print(json.dumps(metrics, indent=4))

    with open('eval_output.json', 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(script_dir)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main()
