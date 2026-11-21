import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from rag_system.evaluation.dataset import load_jsonl, validate_record
from rag_system.evaluation.metrics import abstention_marker, answer_match_metrics, citation_metrics, normalize_answer, numerical_matches, retrieval_metrics

class EvaluationMetricTests(unittest.TestCase):
    def test_retrieval_metrics(self):
        result = retrieval_metrics(["x", "a", "a", "b", "c"], ["a", "b"])
        self.assertEqual(result["recall@1"], 0.0); self.assertEqual(result["recall@3"], 1.0)
        self.assertEqual(result["precision@5"], 0.4); self.assertEqual(result["mrr"], 0.5)
        self.assertGreater(result["ndcg@5"], 0.0)

    def test_numerical_value_and_unit(self):
        self.assertEqual(numerical_matches("Limit: 2.4 MPa", 2.4, "MPa"), {"value_correct": True, "unit_correct": True})
        self.assertEqual(numerical_matches("Limit: 2.4 kPa", 2.4, "MPa"), {"value_correct": True, "unit_correct": False})

    def test_normalization(self):
        self.assertEqual(normalize_answer("  2.4-MPa, MAX! "), "2.4 mpa max")

    def test_answer_scores_ignore_identifier_digits_and_detect_abstention_language(self):
        score = answer_match_metrics("Gauge G-17 was specified.", "Gauge G-17 was specified.")
        self.assertTrue(score["normalized_exact_match"])
        self.assertIsNone(score["numeric_value_match"])
        self.assertIsNone(score["unit_match"])
        self.assertTrue(abstention_marker("There is no information provided about the pump."))

    def test_citation_metrics(self):
        result = citation_metrics(["a", "bad"], ["a", "b"], ["a", "c"])
        self.assertEqual(result, {"citation_precision": 1.0, "citation_coverage": 0.5, "citation_validity": 0.5})

    def test_dataset_schema_and_unanswerable_consistency(self):
        base = {"question_id":"q1","question":"Q?","expected_answer":"A","relevant_document_ids":["d1"],"relevant_evidence_ids":["clause:1"],"category":"direct_factual","answering_allowed":True,"expected_abstention":False}
        validate_record(base)
        with self.assertRaises(ValueError): validate_record({**base,"answering_allowed":False})

    def test_synthetic_ground_truth_integrity_and_distribution(self):
        project=Path(__file__).resolve().parents[1]; fixture=project/"tests/fixtures/synthetic_rag_eval.jsonl"
        corpus=json.loads((project/"tests/fixtures/synthetic_engineering_corpus.json").read_text(encoding="utf-8"))
        rows=load_jsonl(fixture); self.assertEqual(len(rows),30); self.assertEqual(len({x["question_id"] for x in rows}),30)
        counts={name:sum(x["category"]==name for x in rows) for name in {x["category"] for x in rows}}
        self.assertEqual(counts,{"direct_factual":5,"table_based":5,"numerical_unit":5,"multi_chunk":5,"multi_step":5,"unanswerable":5})
        clauses={int(x["number"]) for x in corpus["clauses"]}; tables={int(x["number"]):len(x["rows"]) for x in corpus["tables"]}
        for row in rows:
            self.assertEqual(row["relevant_document_ids"],[corpus["document_id"]])
            for evidence in row["relevant_evidence_ids"]:
                if evidence.startswith("C"): self.assertIn(int(evidence[1:]),clauses)
                elif evidence.startswith("T"):
                    match=__import__("re").fullmatch(r"T(\d+):R(\d+)",evidence); self.assertIsNotNone(match)
                    self.assertLessEqual(int(match.group(2)),tables[int(match.group(1))])
            if row["category"]=="unanswerable": self.assertTrue(row["expected_abstention"])

if __name__=="__main__": unittest.main()
