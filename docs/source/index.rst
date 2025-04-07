.. Evaluation framework for RAG documentation master file, created by
   sphinx-quickstart on Mon Apr  7 17:30:00 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Evaluation framework for RAG documentation
==========================================

This project is an evaluation framework for retrieval-augmented generation (RAG) systems. It provides a set of tools to evaluate the performance of RAG system on chosen datasets. The framework is designed to be flexible and extensible, allowing user to easily configure the evaluation process.

Getting Started
--------------

To run the evaluation pipeline:

.. code-block:: bash

   python -m src.main --corpus=wikitexts --chunker=fixed_size --embedding_model=all-MiniLM-L6-v2 --top_k=5 --reranker=no

The whole setup & running process is explained in details in README.md file on Github repo.

Features
--------

* Support for different text chunking strategies (fixed-size or recursive)
* Multiple embedding models (Sentence Transformers and OpenAI embeddings)
* Cross-encoder reranking for improved results
* Evaluation metrics (IoU, recall, precision)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
