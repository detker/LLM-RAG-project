import re
from typing import Tuple, List, Optional

from fuzzywuzzy import process, fuzz
import tiktoken


class Utils:
    """
    Utility class providing helper functions for text processing and range operations.

    This class contains static methods for range calculations, text search operations,
    and token length calculations, supporting various text processing tasks.
    """

    @staticmethod
    def range_sum(start: int,
                  end: int) -> int:
        """
        Calculate the length of a range (difference between end and start).

        :param start: Starting index of the range
        :type start: int
        :param end: Ending index of the range
        :type end: int
        :return: Length of the range
        :rtype: int
        :raises ValueError: If start is greater than end
        """
        if start > end:
            raise ValueError("start must be less than or equal to end")
        return end - start

    @staticmethod
    def sum_of_ranges(ranges: List[Tuple[int, int]]) -> int:
        """
        Calculate the sum of the lengths of multiple ranges.

        :param ranges: List of tuples, each containing start and end indices
        :type ranges: List[Tuple[int, int]]
        :return: Sum of all range lengths
        :rtype: int
        :raises ValueError: If any range has invalid format
        """
        ret = 0
        try:
            for start, end in ranges:
                ret += Utils.range_sum(start, end)
            return ret
        except Exception as e:
            raise ValueError(f"Invalid range format: {e}")

    @staticmethod
    def union_of_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Merge overlapping ranges into non-overlapping ranges.

        :param ranges: List of tuples, each containing start and end indices
        :type ranges: List[Tuple[int, int]]
        :return: List of non-overlapping merged ranges
        :rtype: List[Tuple[int, int]]
        :raises ValueError: If any range has invalid format
        """
        if not ranges:
            return []
        try:
            sorted_ranges = sorted(ranges, key=lambda x: x[0])
            ret = [sorted_ranges[0]]
            for start, end in sorted_ranges[1:]:
                if start <= ret[-1][1]:
                    ret[-1] = (ret[-1][0], max(ret[-1][1], end))
                else:
                    ret.append((start, end))
            return ret
        except Exception as ex:
            raise ValueError(f"Invalid range format: {ex}")

    @staticmethod
    def intersection_of_ranges(range1: Tuple[int, int],
                               range2: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Find the intersection between two ranges.

        :param range1: First range as (start, end)
        :type range1: Tuple[int, int]
        :param range2: Second range as (start, end)
        :type range2: Tuple[int, int]
        :return: Intersection range or None if no intersection exists
        :rtype: Optional[Tuple[int, int]]
        :raises ValueError: If any range is invalid
        """
        try:
            range1_start, range1_end = range1
            range2_start, range2_end = range2

            if range1_start > range1_end or range2_start > range2_end:
                raise ValueError("Invalid range: start must be less than or equal to end")

            smaller_end = min(range1_end, range2_end)
            bigger_start = max(range1_start, range2_start)

            if bigger_start <= smaller_end:
                return bigger_start, smaller_end
            return None
        except Exception as e:
            raise ValueError(f"Invalid range format: {e}")

    @staticmethod
    def difference_of_ranges(ranges: List[Tuple[int, int]],
                             target: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Remove a target range from a list of ranges.

        :param ranges: List of tuples, each containing start and end indices
        :type ranges: List[Tuple[int, int]]
        :param target: Range to remove as (start, end)
        :type target: Tuple[int, int]
        :return: List of ranges after removing the target range
        :rtype: List[Tuple[int, int]]
        :raises ValueError: If any range is invalid
        """
        if not ranges:
            return []
        try:
            target_start, target_end = target
            if target_start > target_end:
                raise ValueError("Target range: start must be less than or equal to end")
            ret = []

            for start, end in ranges:
                if start > end:
                    raise ValueError("Range: start must be less than or equal to end")
                if start > target_end or end < target_start:
                    ret.append((start, end))
                elif start < target_start and end > target_end:
                    ret.append((start, target_start))
                    ret.append((target_end, end))
                elif start < target_start:
                    ret.append((start, target_start))
                elif end > target_end:
                    ret.append((target_end, end))

            return ret

        except Exception as ex:
            raise ValueError(f"Invalid range format: {ex}")

    @staticmethod
    def find_query_despite_whitespace(document, query):
        """
        Find a query in a document, ignoring differences in whitespace.
        Originally from: https://github.com/brandonstarxel/chunking_evaluation

        :param document: Text to search within
        :type document: str
        :param query: Text to search for
        :type query: str
        :return: Tuple of (matched_text, start_index, end_index) or None if no match
        :rtype: Optional[Tuple[str, int, int]]
        """

        # Normalize spaces and newlines in the query
        normalized_query = re.sub(r'\s+', ' ', query).strip()

        # Create a regex pattern from the normalized query to match any whitespace characters between words
        pattern = r'\s*'.join(re.escape(word) for word in normalized_query.split())

        # Compile the regex to ignore case and search for it in the document
        regex = re.compile(pattern, re.IGNORECASE)
        match = regex.search(document)

        if match:
            return document[match.start(): match.end()], match.start(), match.end()
        else:
            return None

    @staticmethod
    def rigorous_document_search(document: str, target: str):
        """
        Perform a rigorous search of a target string within a document.
        Originally from: https://github.com/brandonstarxel/chunking_evaluation

        This function handles whitespace variations, grammar changes, and minor text alterations.
        It attempts various search strategies in order:
        1. Exact match
        2. Match ignoring whitespace differences
        3. Fuzzy matching by sentence

        :param document: The document to search within
        :type document: str
        :param target: The string to search for
        :type target: str
        :return: Tuple of (matched_text, start_index, end_index) or None if no match
        :rtype: Optional[Tuple[str, int, int]]
        """
        if target.endswith('.'):
            target = target[:-1]

        if target in document:
            start_index = document.find(target)
            end_index = start_index + len(target)
            return target, start_index, end_index
        else:
            raw_search = Utils.find_query_despite_whitespace(document, target)
            if raw_search is not None:
                return raw_search

        # Split the text into sentences
        sentences = re.split(r'[.!?]\s*|\n', document)

        # Find the sentence that matches the query best
        best_match = process.extractOne(target, sentences, scorer=fuzz.token_sort_ratio)

        if best_match[1] < 98:
            return None

        reference = best_match[0]

        start_index = document.find(reference)
        end_index = start_index + len(reference)

        return reference, start_index, end_index

    @staticmethod
    def cl100k_base_length(text: str) -> int:
        """
        Calculate the token length of a text using OpenAI's cl100k_base tokenizer.

        :param text: Text to calculate token length for
        :type text: str
        :return: Number of tokens in the text
        :rtype: int
        :raises RuntimeError: If token calculation fails
        """
        if not text:
            return 0
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(
                enc.encode(
                    text,
                    disallowed_special=(),
                )
            )
        except Exception as ex:
            raise RuntimeError(f"Failed to calculate token length: {ex}")
