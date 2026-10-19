"""
Test Suite for Multi-Sign Offline Reference Validation
======================================================
Ensures reproducibility of the offline validation checks:
1. Confirms all 10 canonical vocabulary labels have at least one valid sample.
2. Confirms all valid samples match exact shape (30, 126) and numerical constraints.
3. Tests that the leave-one-out evaluation executes properly without runtime errors.
4. Confirms that self-matches (dist=0.0) are NOT reported as generalization accuracy.
5. Verifies missing-hand rejection behavior.
"""

import os
import numpy as np
import pytest

from translation.translator import CANONICAL_VOCABULARY
from scripts.validate_multi_reference import (
    audit_reference_files,
    compute_pairwise_distances,
    run_leave_one_out_evaluation
)
from app.web_demo import LiveMultimodalProcessor, DEFAULT_MATCH_THRESHOLD


def test_all_ten_canonical_labels_have_valid_samples():
    """Confirms that all 10 canonical vocabulary signs have valid reference files in data/processed/."""
    records = audit_reference_files()
    valid_records = [r for r in records if r["valid"]]

    recorded_labels = set(r["label"] for r in valid_records)
    for lbl in CANONICAL_VOCABULARY:
        assert lbl in recorded_labels, f"Expected canonical sign '{lbl}' to have at least one valid reference file"


def test_all_reference_samples_conform_to_dimension_and_type():
    """Confirms every reference file has shape (30, 126), dtype float32, no NaNs, and zero blank frames."""
    records = audit_reference_files()
    valid_records = [r for r in records if r["valid"]]
    assert len(valid_records) >= 10

    for r in valid_records:
        assert r["shape"] == (30, 126), f"{r['label']}/{r['file']} shape is {r['shape']}, expected (30, 126)"
        assert r["dtype"] in ("float32", "float64")
        assert not r["has_nan"]
        assert not r["has_inf"]
        assert not r["all_zero"]
        assert r["zero_frames"] == 0, f"{r['label']}/{r['file']} has {r['zero_frames']} blank hand frames"


def test_pairwise_distance_matrix_properties():
    """Confirms symmetry, non-negativity, and zero diagonal for distance matrix."""
    records = [r for r in audit_reference_files() if r["valid"]]
    dist_matrix, sample_ids = compute_pairwise_distances(records)
    n = len(records)

    assert dist_matrix.shape == (n, n)
    # Diagonal must be exactly 0.0 (self-match)
    for i in range(n):
        assert dist_matrix[i, i] == 0.0
    # Symmetric
    assert np.allclose(dist_matrix, dist_matrix.T, atol=1e-5)
    # Non-negative
    assert np.all(dist_matrix >= 0.0)


def test_leave_one_out_excludes_self_matches():
    """Verifies that leave-one-out evaluation excludes self-comparisons."""
    records = [r for r in audit_reference_files() if r["valid"]]
    loo_results = run_leave_one_out_evaluation(records, threshold=DEFAULT_MATCH_THRESHOLD)

    assert len(loo_results) == len(records)
    for res in loo_results:
        # Nearest neighbor must not be the exact test sample
        assert res["nearest_neighbor_id"] != res["test_id"]
        # Nearest distance in leave-one-out is non-zero
        assert res["nearest_distance"] > 0.0


def test_missing_hands_frame_rejection():
    """Verifies that missing hands (zero vectors) report NO MATCH and clear buffer."""
    proc = LiveMultimodalProcessor()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for _ in range(30):
        res = proc.process_frame(blank_frame)

    assert "NO MATCH — HANDS NOT DETECTED" in res[9]
    assert proc.buffer.current_length == 0
    assert res[10] == "..."
    assert res[11] == "..."
