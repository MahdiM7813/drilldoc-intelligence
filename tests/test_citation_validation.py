from __future__ import annotations
import unittest
from rag_system.answering.citation_validation import extract_citations, validate_citations

class CitationValidationTests(unittest.TestCase):
    def setUp(self):
        self.sources = {"S1": {"chunk_id": "a", "page": 81}, "S2": {"chunk_id": "b", "page": 76},
                        "S3": {"chunk_id": "c", "page": 87}, "S4": {"chunk_id": "d", "page": 75}}

    def test_parser_ignores_page_numbers(self):
        self.assertEqual(extract_citations("The answer [S1] Page 81 and [S2]."), ["S1", "S2"])

    def test_valid_ids_are_not_invalid(self):
        result = validate_citations("[S1] and [S2] and [S1]", self.sources)
        self.assertEqual(result.invalid_ids, [])
        self.assertEqual(result.valid_ids, ["S1", "S2"])

    def test_unknown_ids_are_invalid(self):
        result = validate_citations("[S1] [S99]", self.sources)
        self.assertEqual(result.valid_ids, ["S1"])
        self.assertEqual(result.invalid_ids, ["S99"])

    def test_page_claims_are_backend_owned(self):
        result = validate_citations("According to [S1] Page 999", self.sources)
        self.assertEqual(result.sources[0]["page"], 81)
        self.assertEqual(result.invalid_ids, [])

if __name__ == "__main__": unittest.main()
