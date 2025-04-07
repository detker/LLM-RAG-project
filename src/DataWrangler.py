import os
import json
from typing import List, Optional, Tuple, Dict

import pandas as pd

from src.Constants import (DATASET_PATH,
                           CORPORA_EXTENSION,
                           QUESTIONS_EXTENSION)
from src.fixed_token_chunker import FixedTokenChunker
from src.Utils import Utils


class DataWrangler:
    """
    A class for managing corpus data and associated questions.

    This class handles loading corpus text from files, loading question data from CSV files,
    and processing text into chunks for further analysis.
    """

    def __init__(self) -> None:
        """
        Initialize the DataWrangler instance.

        Calls the internal reset method to initialize all attributes.
        """
        self.__reset()

    def __reset(self) -> None:
        """
        Reset all internal attributes to their default values.

        This method is called during initialization and when operations complete.
        """
        self._corpora = None
        self._questions_df = None
        self.corpus_id = None

    def __call__(self) -> (Optional[str], Optional[pd.DataFrame]):
        """
        Return the current corpora and questions data and reset the instance.

        :return: A tuple containing the corpus text and questions DataFrame
        :rtype: tuple[Optional[str], Optional[pd.DataFrame]]
        """
        corpora, questions_df = self._corpora, self._questions_df
        self.__reset()
        return corpora, questions_df

    def get_corpora(self) -> Optional[List[str]]:
        """
        Get the currently loaded corpus text.

        :return: The corpus text or None if not loaded
        :rtype: Optional[List[str]]
        """
        return self._corpora

    def get_questions_df(self) -> Optional[pd.DataFrame]:
        """
        Get the currently loaded questions DataFrame.

        :return: The questions DataFrame or None if not loaded
        :rtype: Optional[pd.DataFrame]
        """
        return self._questions_df

    def _load_corpora_file(self,
                           corpus_id: str) -> str:
        """
        Load the corpus text from a file.

        :param corpus_id: Identifier for the corpus to load
        :type corpus_id: str
        :return: The loaded corpus text
        :rtype: str
        :raises FileNotFoundError: If the corpus file doesn't exist
        :raises ValueError: If the corpus file is empty
        :raises Exception: If there's an error reading the file
        """
        path = os.path.join(DATASET_PATH, corpus_id + CORPORA_EXTENSION)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Corpora file '{path}' not found.")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                res = f.read()
            if not res:
                raise ValueError(f"Corpora file '{path}' is empty.")
            return res
        except Exception as ex:
            raise Exception(f"Failed to read corpus file: {ex}")

    def _find_questions_file(self) -> str:
        """
        Find the questions file in the dataset directory.

        :return: Path to the questions file
        :rtype: str
        :raises FileNotFoundError: If no questions file is found
        :raises ValueError: If multiple questions files are found
        """
        path = None
        try:
            for file in os.listdir(DATASET_PATH):
                if file.endswith(QUESTIONS_EXTENSION):
                    if path is not None:
                        raise ValueError(f'Multiple question {QUESTIONS_EXTENSION} found.')
                    path = os.path.join(DATASET_PATH, file)
            if path is None:
                raise FileNotFoundError(f'No question {QUESTIONS_EXTENSION} found.')

            return path
        except Exception as ex:
            raise FileNotFoundError(f"Failed to locate questions file: {ex}")

    def load_corpora_and_questions(self,
                                   corpus_id: str) -> None:
        """
        Load corpus text and associated questions.

        :param corpus_id: Identifier for the corpus to load
        :type corpus_id: str
        :raises ValueError: If corpus_id is empty
        :raises RuntimeError: If loading fails
        """
        if not corpus_id:
            raise ValueError("'corpus_id' cannot be None")

        self.corpus_id = corpus_id
        try:
            self._corpora = self._load_corpora_file(corpus_id)
            questions_path = self._find_questions_file()
            self._questions_df = pd.read_csv(questions_path)
            self.__clear_questions()
            self._convert_to_json()
        except Exception as ex:
            self.__reset()
            raise RuntimeError(f"Failed to load corpora or questions: {ex}")

    def _convert_to_json(self):
        """
        Convert the 'references' column in the questions DataFrame from JSON strings to Python objects.
        """
        def json_loads(row):
            if pd.isna(row):
                return None
            try:
                return json.loads(row)
            except json.JSONDecodeError:
                pass

        self._questions_df['references'] = self._questions_df['references'].apply(json_loads)

    def __clear_questions(self) -> None:
        """
        Filter the questions DataFrame to only include rows for the current corpus.

        :raises KeyError: If the 'corpus_id' column is not found in the DataFrame
        """
        try:
            self._questions_df = self._questions_df[self._questions_df['corpus_id'] == self.corpus_id]
        except KeyError as ex:
            raise KeyError(f"Column 'corpus_id' not found in questions DataFrame: {ex}")

    def construct_chunks_and_meta(self,
                                  chunker: FixedTokenChunker) -> Tuple[List[str], List[Dict[str, int]], pd.DataFrame]:
        """
        Split the corpus into chunks and generate metadata.

        :param chunker: A chunker object to split the text
        :type chunker: FixedTokenChunker
        :return: A tuple containing chunks, chunk metadata, and the questions DataFrame
        :rtype: tuple[List[str], List[Dict[str, int]], pd.DataFrame]
        :raises ValueError: If corpora or questions are not loaded
        :raises RuntimeError: If chunk processing fails
        """
        if self._corpora is None or self._questions_df is None:
            raise ValueError("Corpora or questions not loaded. Call load_corpora_and_questions function first.")

        try:
            chunks = chunker.split_text(self._corpora)
            chunk_metas = []
            for chunk in chunks:
                _, start_idx, end_idx = Utils.rigorous_document_search(self._corpora, chunk)
                chunk_metas.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                })
            questions_df = self._questions_df
            self.__reset()
            return chunks, chunk_metas, questions_df
        except (ValueError, IndexError) as ex:
            raise RuntimeError(f"Failed to process chunks: {ex}")
