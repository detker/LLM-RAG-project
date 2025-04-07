# Intelligent Chunking Methods for Code Documentation RAG - Jetbrains

## 📋 Table of Contents

- [Overview](#overview)
  - [Datasets](#datasets)
  - [Chunking Algorithms](#chunking-algorithms)
  - [Retrieval Metrics](#retrieval-metrics)
  - [Embedding Models](#embedding-models)
  - [Retrieved Top-k Values](#retrieved-top-k-values)
  - [Reranking](#reranking)
- [Setup & Usage](#setup--usage)
  - [Configuration](#configuration)
  - [Usage](#usage)
  - [Running the Evaluation - Example](#running-the-evaluation---example)
  - [Building Docs](#building-docs)
- [Experiments](#experiments)


## 🔎 Overview
This project is an evaluation framework for retrieval-augmented generation (RAG) systems.
It provides a set of tools to evaluate the performance of RAG systems on chosen datasets.
The framework is designed to be flexible and extensible, allowing users to easily configure the evaluation process.

### Datasets

With the framework comes example datasets,

- `chatlogs.md`: Subset (first 7,727 tokens) of The UltraChat 200k dataset, consisting of high-quality 1.4M dialogues generated by ChatGPT.
- `state_of_the_union.md`: A plain, clean transcript of the State of the Union Address in 2024 (10,444 tokens long).
- `wikitexts.md`: Subset (first 26,649 tokens) of over 100 million tokens from verified articles on Wikipedia.

available to clone through the `clone_data.sh` shell script:

```bash
chmod +x clone_data.sh
./clone_data.sh
```

After script execution, datasets are stored in the `data` directory and can be used for evaluation purposes.

In addition, example datasets come with example queries and information about where in the corpora the answers can be found, all contained in a single `questions_df.csv` file.
Example datasets contain 56, 76, and 144 questions respectively.

Example record from `questions_df.csv`:


| question                                                                                                                                                 | references                                                                                                                                                                                                                                                                                                                                                           | corpus_id          |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| What significant regulatory changes and proposals has President Biden's administration implemented or announced regarding fees and pricing transparency? | `[{"content": "My administration announced we're cutting credit card late fees from $32 to $8.", "start_index": 27346, "end_index": 27425}, {"content": "My administration has proposed rules to make cable, travel, utilities, and online ticket sellers tell you the total price up front so there are no surprises.", "start_index": 27866, "end_index": 28023}]` | state_of_the_union |

It's important to note that if users want to use their own datasets, they need to provide a `.csv` file with the same structure as the one provided for example datasets.

### Chunking Algorithms

The framework provides two chunking algorithms to split the documents into smaller chunks:

- `FixedTokenChunker`: Splits the document into fixed-size chunks based on the number of tokens.
- `RecursiveTokenChunker`: Splits the document into chunks based on the number of tokens but also uses a recursive approach to find the best chunking points in the text based on provided separators.

These strategies are among the most popular ones, often used by many RAG systems.
Both chunking algorithms do not take into account the semantic meaning of the text.

The implementations of both chunking algorithms are taken from [here](https://github.com/brandonstarxel/chunking_evaluation).

### Retrieval Metrics

The framework provides a set of retrieval metrics to evaluate the performance of RAG systems. For each question $q$, the following metrics are calculated as functions of chunked corpus $C$:

- `Precision`: $Precision_{q}(C) = \frac{|t_e \cap t_r|}{|t_r|}$
- `Recall`: $Recall_{q}(C) = \frac{|t_e \cap t_r|}{|t_e|}$
- `Intersection over Union (IoU)`: $IoU_{q}(C) = \frac{|t_e \cap t_r|}{|t_e \cup t_r|}$

where $|t_e|$ is the number of tokens in the expected answer, $|t_r|$ is the number of tokens in the retrieved answer, and $|t_e \cap t_r|$ is the number of tokens in the intersection of expected and retrieved answers.
The metrics are calculated for each question and then averaged over all questions from the dataset.

Precision helps evaluate if the retrieval system is pulling in unnecessary documents that waste context space, 
Recall helps evaluate if the retrieval system is missing relevant documents, and the latter penalizes not only missing relevant documents but also pulling in unnecessary documents.

### Embedding Models
The framework provides four embedding models to evaluate the performance of RAG systems:
- Two Open Source Embedding Models from the `sentence-transformers` library, available through HuggingFace:
  - `all-MiniLM-L6-v2`
  - `multi-qa-mpnet-base-dot-v1`
- Two OpenAI's Embedding Models:
    - `text-embedding-3-small`
    - `text-embedding-3-large`

For the latter category, users need to provide their own `OPENAI_API_KEY` ([See Configuration section](#configuration)) in order to use the models.

### Retrieved Top-k Values
The framework provides a set of `top-k` values to evaluate the performance of RAG systems. The user can choose from the following values:
- `1`
- `3`
- `5`
- `10`

The `top-k` values are used to retrieve the most relevant chunks from the corpus.
Additionally, there's a `Min` option available, which retrieves the minimum number of chunks that were relevant to the question, bounded from above by the `top-k` value (set through `src/Constants.py` ([see Configuration section](#configuration))).

### Reranking
The framework additionally provides a reranking algorithm.
Reranking performs a second pass over the `top_k`*`Constants.RERANKER_MULTIPLIER` retrieved chunks to find the most relevant `top_k` ones by inferencing OpenAI's `gpt-4o-mini` model.


## ⚙️ Setup & Usage
To set up the project, you need to have `Python 3.11.4` and `Conda 23.7.3` installed on your machine.

```bash
git clone https://github.com/detker/LLM-RAG-project
cd LLM-RAG-project
conda create -n llm_rag python=3.11.4
conda activate llm_rag
pip install -r requirements.txt
```

### Configuration
Hyperparameters (e.g. `CHUNK_SIZE`, `CHUNK_OVERLAP`) are stored in the `Constants.py` file, which is located in the `src` directory. Feel free to change those parameters to fit your needs.
The `OPENAI_API_KEY` constant is also located in that file. Below is the table with all constants located in `Constants.py`:

| Constant | Default Value   | Description                                                               |
|----------|-----------------|---------------------------------------------------------------------------|
| `DATASET_PATH` | `'data'`        | Root directory for dataset files                                          |
| `CORPORA_EXTENSION` | `'.md'`         | File extension for corpus documents                                       |
| `QUESTIONS_EXTENSION` | `'.csv'`        | File extension for questions files                                        |
| `QUESTIONS_FILENAME` | `'questions_df'` | Base filename for questions dataset                                       |
| `CHUNK_SIZE` | `200`           | Number of tokens in each chunk using fixed-size chunking                  |
| `CHUNK_OVERLAP` | `0`             | Number of overlapping tokens between consecutive chunks                   |
| `MINIMALIZE_CHUNKS` | `False`         | Flag to enable chunk size minimization                                    |
| `EMBD_BATCH_SIZE` | `32`            | Batch size for processing embeddings                                      |
| `OPENAI_API_KEY` | `None`          | API key for OpenAI services                                               |
| `RERANKER_MULTIPLIER` | `3`             | Multiplier for the number of chunks to rerank                                 |
| `MAX_WORKERS` | `5`             | Maximum number of concurrent workers for parallel processing API requests |

### Usage

First, we have to obtain the datasets.

##### Windows:

Download datasets and questions, place them in the `data` directory:
- [Chatlogs](https://raw.githubusercontent.com/brandonstarxel/chunking_evaluation/refs/heads/main/chunking_evaluation/evaluation_framework/general_evaluation_data/corpora/chatlogs.md)
- [State of The Union](https://raw.githubusercontent.com/brandonstarxel/chunking_evaluation/refs/heads/main/chunking_evaluation/evaluation_framework/general_evaluation_data/corpora/state_of_the_union.md)
- [Wikitexts](https://raw.githubusercontent.com/brandonstarxel/chunking_evaluation/refs/heads/main/chunking_evaluation/evaluation_framework/general_evaluation_data/corpora/wikitexts.md)
- [questions_df.csv](https://raw.githubusercontent.com/brandonstarxel/chunking_evaluation/refs/heads/main/chunking_evaluation/evaluation_framework/general_evaluation_data/questions_df.csv)

##### Linux/MacOS:
On Linux/MacOS the framework provides a user-friendly shell script `clone_data.sh` to download the datasets and questions and 
place them in the `data` directory automatically.

```bash
chmod +x clone_data.sh
./clone_data.sh
```

Then, regarding the used operating system, we can run the evaluation script from the project directory:

```bash
python -m src.main \
    --corpus=<dataset_name> \
    --chunker=<chunking_strategy> \
    --embedding_model=<model_name> \
    --top_k=<k_value> \
    --reranker=<yes|no>
```

where:
- `dataset_name`: One of `chatlogs`, `wikitexts`, `state_of_the_union`
- `chunking_strategy`: One of `fixed_size`, `recursive`
- `model_name`: One of `all-MiniLM-L6-v2`, `multi-qa-mpnet-base-dot-v1`, `text-embedding-3-small`, `text-embedding-3-large`
- `k_value`: Number of top chunks to retrieve (1, 3, 5, or 10)
- `reranker`: Whether to enable reranking (`yes` or `no`)

Once again, **when on Linux/MacOS** the framework provides a user-friendly shell script `run.sh` to run the evaluation. Full execution for **Linux/MacOS** machines is provided below:
```bash
chmod +x clone_data.sh
./clone_data.sh
chmod +x run.sh
./run.sh
```

Single evaluation output is shown on standard output and also saved in the `eval_output.json` file.

### Running the Evaluation - Example
```bash
>> ./run.sh
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃     RAG Evaluation Pipeline - Configuration      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

1) chatlogs
2) wikitexts
3) state_of_the_union
Select a dataset: 2
1) fixed_size
2) recursive
Select a chunking strategy: 1
1) all-MiniLM-L6-v2	       3) text-embedding-3-small
2) multi-qa-mpnet-base-dot-v1  4) text-embedding-3-large
Select an embedding model: 1
1) 1
2) 3
3) 5
4) 10
Select top_k value: 3
1) yes
2) no
Enable reranker? (yes/no): 2

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          Selected Configuration Options          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
Dataset:           wikitexts
Chunking strategy: fixed_size
Embedding model:   all-MiniLM-L6-v2
Top_k:             5
Reranker:          no
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Are you sure? [Y/n] Y

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃              Executing Evaluation                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
▶ Working directory: /Users/detker/jetbrainsRAG
▶ Start time: 01:09:57
───────────────────────────────────────────────────
Running evaluation with the following command:
python -m src.main \
    --corpus="wikitexts" \
    --chunker="fixed_size" \
    --embedding_model="all-MiniLM-L6-v2" \
    --top_k="5" \
    --reranker="no"
───────────────────────────────────────────────────

{
    "iou_score": 0.04512450191355461,
    "recall_score": 0.7976012175959286,
    "precision_score": 0.04538893325419158
}

───────────────────────────────────────────────────
✓ Evaluation completed successfully!
▶ End time: 01:10:05
```

### Building Docs
To build the documentation, first you need to have `Sphinx` installed. You can do this by running:

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
```

Then, you can build the documentation by running:
```bash
cd docs
make html
```

You can display the documentation by opening the `index.html` file in the `./docs/build/html` directory.

## 🧪 Experiments
The framework provides a set of experiments that differ by configuration parameters, evaluating the performance of RAG systems on the `wikitexts` example corpus. 
The experiments are conducted in `.ipynb` notebooks stored in the `notebooks` directory, and the results are stored in the `output` directory as `.csv` files.

The notebooks should provide a walkthrough of the experiments (`experiments.ipynb`), the results, and the analysis of the results (`experiments_st.ipynb`).
In addition, the `experiments_gpt.ipynb` notebook provides an analysis of using OpenAI's `text-embedding-3-large` model and the reranking procedure performed on the most promising configurations of parameters.

Therefore, the preferred order of notebook execution is:
1. `experiments.ipynb`
2. `experiments_st.ipynb`
3. `experiments_gpt.ipynb`