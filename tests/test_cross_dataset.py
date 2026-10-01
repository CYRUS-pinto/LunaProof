"""
Tests for Cross-Dataset Generalization Evaluator
"""
import pytest
from lunaproof.cross_dataset import CrossDatasetEvaluator

def test_cross_dataset_evaluator():
    evaluator = CrossDatasetEvaluator(seed=42)
    res = evaluator.evaluate_cross_dataset()
    assert res["status"] in ("PASS", "FAIL")
    assert "dataset_a_equatorial" in res
    assert "dataset_b_south_pole" in res
    assert 0.0 <= res["generalization_score"] <= 1.0

