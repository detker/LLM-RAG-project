from typing import Dict, List, Union

import numpy as np
import pandas as pd

from src.Utils import Utils


class RetrievalEvaluator:
    """
    A class that evaluates the performance of text chunk retrievals against reference data.

    This evaluator calculates standard information retrieval metrics including
    Intersection over Union (IoU), recall, and precision scores for text chunks
    retrieved from a document relative to reference passages.
    """

    @staticmethod
    def evaluate_retrievals(chunks_metadatas: np.ndarray,
                            questions_df: pd.DataFrame,
                            min_top_k: bool = False) -> Dict[str, List[Union[float, int]]]:
        """
        Evaluate retrieval performance by comparing retrieved chunks with reference data.

        :param chunks_metadatas: Array of chunk metadata containing start/end indices
        :type chunks_metadatas: np.ndarray
        :param questions_df: DataFrame containing questions and their reference passages
        :type questions_df: pd.DataFrame
        :param min_top_k: Whether to use only the minimum number of chunks needed (default: False)
        :type min_top_k: bool
        :return: Dictionary containing lists of IoU, recall, and precision scores
        :rtype: Dict[str, List[Union[float, int]]]
        :raises ValueError: If input data is empty or has mismatched lengths
        :raises RuntimeError: If evaluation fails
        """
        if chunks_metadatas is None or len(chunks_metadatas) == 0:
            raise ValueError("chunks_metadatas cannot be empty")
        if questions_df.empty:
            raise ValueError("questions_df cannot be empty")
        if len(chunks_metadatas) != len(questions_df):
            raise ValueError("Length mismatch between chunks_metadatas and questions_df")

        iou_scores = []
        recall_scores = []
        precision_scores = []
        highlighted_chunks_counts = []

        try:
            for (index, row), chunk_metadata in zip(questions_df.iterrows(), chunks_metadatas):
                try:
                    references = row['references']
                    if not references:
                        print(f"Skipping row {index}: empty references")
                        continue

                    metrics = RetrievalEvaluator._calculate_all_metrics(chunk_metadata, references)

                    iou_scores.append(metrics['iou'])
                    recall_scores.append(metrics['recall'])
                    precision_scores.append(metrics['precision'])
                    highlighted_chunks_counts.append(metrics['highlighted_chunks'])
                except Exception as ex:
                    print(f"Error processing row {index}: {ex}")

                    continue

            if min_top_k:
                if len(highlighted_chunks_counts) != len(questions_df):
                    raise ValueError("Length mismatch between highlighted_chunks_counts and questions_df")
                iou_scores = []
                recall_scores = []
                precision_scores = []
                for (index, row), highlighted_chunks_count, chunk_metadata in zip(questions_df.iterrows(),
                                                                                  highlighted_chunks_counts,
                                                                                  chunks_metadatas):
                    try:
                        references = row['references']
                        metrics = RetrievalEvaluator._calculate_all_metrics(chunk_metadata, references,
                                                                            highlighted_chunks_count)

                        iou_scores.append(metrics['iou'])
                        recall_scores.append(metrics['recall'])
                        precision_scores.append(metrics['precision'])
                    except Exception as ex:
                        print(f"Error in min_top_k processing for row {index}: {ex}")
                        continue

            return {
                'iou_score': iou_scores,
                'recall_score': recall_scores,
                'precision_score': precision_scores,
            }

        except Exception as e:
            raise RuntimeError(f"Failed to evaluate retrievals: {e}")

    @staticmethod
    def _calculate_all_metrics(chunk_metadatas: np.ndarray,
                               references: List[Dict[str, Union[str, int]]],
                               highlighted_chunks_count: int = None) -> Dict[str, Union[float, int]]:
        """
        Calculate evaluation metrics for a single question/retrieval pair.

        This method computes IoU, recall, precision, and the number of highlighted chunks
        by comparing chunk metadata with reference passage data.

        :param chunk_metadatas: Array of metadata for retrieved chunks
        :type chunk_metadatas: np.ndarray
        :param references: List of reference passage dictionaries with start_index and end_index
        :type references: List[Dict[str, Union[str, int]]]
        :param highlighted_chunks_count: Limit calculation to this number of top chunks (default: None, uses all chunks)
        :type highlighted_chunks_count: int, optional
        :return: Dictionary containing calculated metrics (iou, recall, precision, highlighted_chunks)
        :rtype: Dict[str, Union[float, int]]
        :raises RuntimeError: If metric calculation fails
        """
        if highlighted_chunks_count is None:
            highlighted_chunks_count = len(chunk_metadatas)

        try:
            reference_ranges = [(ref['start_index'], ref['end_index']) for ref in references]
            unused_highlights = reference_ranges.copy()
            numerator_sets = []
            all_chunk_ranges = []
            relevant_chunk_ranges = []
            highlighted_chunks = 0

            for chunk in chunk_metadatas[:highlighted_chunks_count]:
                chunk_start, chunk_end = chunk['chunk_meta']['start_idx'], chunk['chunk_meta']['end_idx']
                chunk_range = (chunk_start, chunk_end)
                all_chunk_ranges.append(chunk_range)

                has_intersection = False

                for ref_range in reference_ranges:
                    intersection = Utils.intersection_of_ranges(chunk_range, ref_range)

                    if intersection:
                        has_intersection = True
                        numerator_sets = Utils.union_of_ranges([intersection] + numerator_sets)

                        unused_highlights = Utils.difference_of_ranges(unused_highlights, intersection)

                if has_intersection:
                    highlighted_chunks += 1
                    relevant_chunk_ranges.append(chunk_range)

            if numerator_sets:
                numerator_value = Utils.sum_of_ranges(numerator_sets)
            else:
                numerator_value = 0

            recall_denominator = Utils.sum_of_ranges(reference_ranges)
            precision_denominator = Utils.sum_of_ranges(all_chunk_ranges)
            iou_denominator = precision_denominator + Utils.sum_of_ranges(unused_highlights)

            recall = numerator_value / recall_denominator if recall_denominator > 0 else 0.0
            precision = numerator_value / precision_denominator if precision_denominator > 0 else 0.0
            iou = numerator_value / iou_denominator if iou_denominator > 0 else 0.0

            return {
                'iou': iou,
                'recall': recall,
                'precision': precision,
                'highlighted_chunks': highlighted_chunks
            }

        except Exception as ex:
            raise RuntimeError(f"Error calculating metrics: {ex}")
