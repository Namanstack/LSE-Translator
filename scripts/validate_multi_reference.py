"""
LSE Translator - Comprehensive Offline Multi-Reference Validation
==================================================================
Performs scientific audit and statistical validation of all 10 saved
reference sign samples in data/processed/.

Validates:
1. File compatibility, keys, and shape (30, 126).
2. Spatio-temporal normalization and numerical integrity.
3. Full pairwise distance matrix (inter-class and intra-class).
4. Systematic reference bias inspection (identifies if any sign acts as an attractor).
5. Leave-One-Out validation on multi-sample signs (GRACIAS, ADIOS, POR_FAVOR).
6. Honest separability analysis (distinguishing trivial self-matches from true generalization).
7. Threshold behavior and missing-hand rejection.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Any

# Ensure project root is available
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.web_demo import LiveMultimodalProcessor, compute_sequence_distance, DEFAULT_MATCH_THRESHOLD
from translation.translator import CANONICAL_VOCABULARY, LSETranslator


def audit_reference_files() -> List[Dict[str, Any]]:
    """Audits each .npz reference file across all 10 canonical vocabulary directories."""
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    audit_records = []

    for label in CANONICAL_VOCABULARY:
        label_dir = os.path.join(processed_dir, label)
        if not os.path.isdir(label_dir):
            audit_records.append({
                "label": label,
                "file": "MISSING_DIR",
                "valid": False,
                "error": "Directory does not exist"
            })
            continue

        npz_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".npz") and not f.startswith("face_")])
        if not npz_files:
            audit_records.append({
                "label": label,
                "file": "NO_SAMPLES",
                "valid": False,
                "error": "No valid hand .npz samples in directory"
            })
            continue

        for fname in npz_files:
            fpath = os.path.join(label_dir, fname)
            rec = {
                "label": label,
                "file": fname,
                "path": fpath,
                "valid": False,
                "error": None
            }
            try:
                data = np.load(fpath, allow_pickle=True)
                rec["keys"] = list(data.keys())
                if "features" not in data:
                    rec["error"] = "Missing 'features' key"
                    audit_records.append(rec)
                    continue

                feat = data["features"]
                rec["shape"] = tuple(feat.shape)
                rec["dtype"] = str(feat.dtype)
                rec["has_nan"] = bool(np.isnan(feat).any())
                rec["has_inf"] = bool(np.isinf(feat).any())
                rec["all_zero"] = bool(np.all(feat == 0))
                rec["zero_frames"] = int(np.sum(np.all(feat == 0, axis=1)))
                rec["signer"] = str(data["signer"]) if "signer" in data else "unknown"
                rec["recording_id"] = str(data["recording_id"]) if "recording_id" in data else "unknown"

                # Validity rules
                if rec["shape"] != (30, 126):
                    rec["error"] = f"Invalid shape {rec['shape']}, expected (30, 126)"
                elif rec["has_nan"] or rec["has_inf"]:
                    rec["error"] = "Contains NaN or Inf"
                elif rec["all_zero"]:
                    rec["error"] = "All features are zero"
                elif rec["zero_frames"] > 0:
                    rec["error"] = f"Contains {rec['zero_frames']} blank hand frames"
                else:
                    rec["valid"] = True
                    rec["features"] = feat.astype(np.float32)

            except Exception as e:
                rec["error"] = f"Exception loading file: {e}"

            audit_records.append(rec)

    return audit_records


def compute_pairwise_distances(valid_samples: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    """Computes pairwise MAE distance matrix between all valid samples."""
    n = len(valid_samples)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    sample_ids = [f"{s['label']}/{s['file']}" for s in valid_samples]

    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = compute_sequence_distance(
                valid_samples[i]["features"],
                valid_samples[j]["features"]
            )

    return dist_matrix, sample_ids


def run_leave_one_out_evaluation(valid_samples: List[Dict[str, Any]], threshold: float = DEFAULT_MATCH_THRESHOLD):
    """
    Evaluates each sample against the rest of the library excluding itself.
    This separates trivial self-matches (dist=0.0) from genuine nearest-neighbor retrieval.
    """
    results = []
    n = len(valid_samples)

    for i in range(n):
        test_sample = valid_samples[i]
        test_feat = test_sample["features"]
        true_label = test_sample["label"]
        test_id = f"{true_label}/{test_sample['file']}"

        # Count how many other samples of this label exist
        same_label_count = sum(1 for j in range(n) if j != i and valid_samples[j]["label"] == true_label)

        best_label = None
        best_dist = float("inf")
        best_id = None
        same_sign_min_dist = float("inf")
        same_sign_best_id = None
        diff_sign_min_dist = float("inf")
        diff_sign_best_id = None

        for j in range(n):
            if i == j:
                continue  # EXCLUDE SELF

            cand = valid_samples[j]
            cand_id = f"{cand['label']}/{cand['file']}"
            dist = compute_sequence_distance(test_feat, cand["features"])

            if dist < best_dist:
                best_dist = dist
                best_label = cand["label"]
                best_id = cand_id

            if cand["label"] == true_label:
                if dist < same_sign_min_dist:
                    same_sign_min_dist = dist
                    same_sign_best_id = cand_id
            else:
                if dist < diff_sign_min_dist:
                    diff_sign_min_dist = dist
                    diff_sign_best_id = cand_id

        is_match = (best_dist <= threshold)
        correct_retrieval = (best_label == true_label) if same_label_count > 0 else None

        results.append({
            "test_id": test_id,
            "true_label": true_label,
            "same_label_count": same_label_count,
            "nearest_neighbor_id": best_id,
            "nearest_neighbor_label": best_label,
            "nearest_distance": best_dist,
            "is_within_threshold": is_match,
            "correct_retrieval": correct_retrieval,
            "same_sign_min_dist": same_sign_min_dist if same_label_count > 0 else None,
            "diff_sign_min_dist": diff_sign_min_dist,
            "diff_sign_best_id": diff_sign_best_id
        })

    return results


def main():
    print("=" * 80)
    print(" LSE TRANSLATOR — OFFLINE MULTI-SIGN REFERENCE VALIDATION")
    print(f" Target Vocabulary: {len(CANONICAL_VOCABULARY)} Canonical Signs")
    print(f" Match Distance Threshold: {DEFAULT_MATCH_THRESHOLD:.2f} (MAE)")
    print("=" * 80)

    # 1. Audit Reference Files
    print("\n--- 1. FILE INTEGRITY & DIMENSIONALITY AUDIT ---")
    records = audit_reference_files()
    valid_samples = [r for r in records if r["valid"]]

    for r in records:
        status = "PASS" if r["valid"] else "FAIL"
        extra = f"shape={r.get('shape')}, zero_frames={r.get('zero_frames')}" if r["valid"] else f"error={r['error']}"
        print(f"[{status}] {r['label']:<10} {r['file']:<20} | {extra}")

    print(f"\nTotal Reference Files Found: {len(records)}")
    print(f"Valid Reference Sequences:   {len(valid_samples)}/{len(records)}")

    labels_with_samples = sorted(list(set(r["label"] for r in valid_samples)))
    print(f"Vocabulary Coverage:         {len(labels_with_samples)}/{len(CANONICAL_VOCABULARY)} signs recorded")
    print(f"Active Signs:                {', '.join(labels_with_samples)}")

    if len(valid_samples) < len(CANONICAL_VOCABULARY):
        print("\n[WARNING] Not all 10 signs have valid reference samples!")
        return

    # 1b. Handedness and Active Landmark Analysis
    print("\n--- 1b. HANDEDNESS & MODALITY ANALYSIS ---")
    for s in valid_samples:
        feat = s["features"]
        left_hand = feat[:, :63]
        right_hand = feat[:, 63:]
        left_active = not np.all(left_hand == 0)
        right_active = not np.all(right_hand == 0)
        left_norm = float(np.linalg.norm(left_hand))
        right_norm = float(np.linalg.norm(right_hand))
        mod_desc = "Dual-Hand" if (left_active and right_active) else ("Right-Hand Only" if right_active else ("Left-Hand Only" if left_active else "No-Hands"))
        print(f"[{mod_desc:<16}] {s['label']:<10} {s['file']:<18} | Left norm: {left_norm:5.1f} | Right norm: {right_norm:5.1f}")

    # 2. Pairwise Distance Matrix
    print("\n--- 2. PAIRWISE MAE DISTANCE MATRIX ---")
    dist_matrix, sample_ids = compute_pairwise_distances(valid_samples)
    n = len(valid_samples)

    # Full sample headers for table printing
    print(f"Index Mapping:")
    for idx, sid in enumerate(sample_ids):
        print(f"  [{idx:02d}] {sid}")

    print("\nDistance Matrix (row vs col index):")
    idx_header = "     " + "".join([f"[{j:02d}]  " for j in range(n)])
    print(idx_header)
    print("-" * len(idx_header))
    for i in range(n):
        row_str = f"[{i:02d}] " + "".join([f"{dist_matrix[i, j]:6.3f} " for j in range(n)])
        print(row_str)

    # 3. Systematic Bias / Centroid Attractor Check
    print("\n--- 3. SYSTEMATIC BIAS & ATTRACTOR INSPECTION ---")
    # For each sample, compute its average distance to all DIFFERENT signs
    mean_inter_dists = {}
    for i, s in enumerate(valid_samples):
        diff_dists = [dist_matrix[i, j] for j in range(n) if valid_samples[j]["label"] != s["label"]]
        mean_inter_dists[sample_ids[i]] = float(np.mean(diff_dists))

    sorted_bias = sorted(mean_inter_dists.items(), key=lambda x: x[1])
    print("Mean MAE distance to other signs (lower means closer to the average landmark cluster):")
    for sid, mean_d in sorted_bias:
        print(f"  {sid:<25}: mean distance to others = {mean_d:.3f}")

    closest_attractor = sorted_bias[0]
    print(f"\nPotential attractor candidate: {closest_attractor[0]} (mean dist {closest_attractor[1]:.3f})")
    if closest_attractor[1] < DEFAULT_MATCH_THRESHOLD:
        print("  [ALERT] Mean inter-class distance is BELOW match threshold! High risk of false positive collisions.")
    else:
        print(f"  [CONFIRMED] Mean inter-class distance ({closest_attractor[1]:.3f}) is well above match threshold ({DEFAULT_MATCH_THRESHOLD:.2f}).")

    # 4. Leave-One-Out Evaluation (Separating Self-Matches from Independent Generalization)
    print("\n--- 4. NEAREST NEIGHBOR & SEPARABILITY EVALUATION ---")
    loo_results = run_leave_one_out_evaluation(valid_samples, threshold=DEFAULT_MATCH_THRESHOLD)

    print(f"{'Sample ID':<24} | {'Held-out?':<9} | {'Nearest Other':<22} | {'Dist':<6} | {'Margin (Inter-Intra)':<20} | {'Status'}")
    print("-" * 105)

    multi_sample_evals = []
    single_sample_evals = []

    for res in loo_results:
        sid = res["test_id"]
        if res["same_label_count"] > 0:
            multi_sample_evals.append(res)
            held_out = "YES (>=2)"
            margin = res["diff_sign_min_dist"] - res["same_sign_min_dist"]
            margin_str = f"+{margin:.3f}" if margin > 0 else f"{margin:.3f}"
            status = "CORRECT PAIR" if res["correct_retrieval"] else "MISCLASSIFIED"
            nearest_str = f"{res['nearest_neighbor_id']} ({res['nearest_distance']:.3f})"
            print(f"{sid:<24} | {held_out:<9} | {nearest_str:<22} | {res['nearest_distance']:<6.3f} | {margin_str:<20} | {status}")
        else:
            single_sample_evals.append(res)
            held_out = "NO (1 only)"
            nearest_diff = f"{res['diff_sign_best_id']} ({res['diff_sign_min_dist']:.3f})"
            status = "NO OVERLAP" if res["diff_sign_min_dist"] > DEFAULT_MATCH_THRESHOLD else "COLLISION RISK"
            print(f"{sid:<24} | {held_out:<9} | {nearest_diff:<22} | {res['diff_sign_min_dist']:<6.3f} | {'N/A (single sample)':<20} | {status}")

    # 5. Independent Generalization Summary
    print("\n--- 5. INDEPENDENT RECOGNITION VALIDATION SUMMARY ---")
    if multi_sample_evals:
        correct_multi = sum(1 for r in multi_sample_evals if r["correct_retrieval"])
        total_multi = len(multi_sample_evals)
        print(f"Multi-sample Leave-One-Out Retrieval: {correct_multi}/{total_multi} ({correct_multi/total_multi*100:.1f}%)")
        for r in multi_sample_evals:
            print(f"  * {r['test_id']} -> Nearest neighbor is {r['nearest_neighbor_id']} (dist: {r['nearest_distance']:.3f})")
    else:
        print("No multi-sample signs available for leave-one-out testing.")

    print(f"\nSingle-sample Signs: {len(single_sample_evals)}/10")
    print("  * For single-sample signs (HOLA, SI, NO, AGUA, AYUDA, PERDON, BIEN):")
    print("    Self-match distance is 0.000 (trivially matching identical vector in memory).")
    print("    CRITICAL SCIENTIFIC REPORTING: True accuracy across unseen gestures is NOT YET ESTABLISHED")
    print("    due to lack of independent test samples.")

    # 6. Inter-Class Margin Distribution
    all_inter_dists = [dist_matrix[i, j] for i in range(n) for j in range(n) if valid_samples[i]["label"] != valid_samples[j]["label"]]
    print(f"\n--- 6. STATISTICAL INTER-CLASS DISTANCE DISTRIBUTION ---")
    print(f"Minimum Inter-Class Distance: {np.min(all_inter_dists):.3f}")
    print(f"Maximum Inter-Class Distance: {np.max(all_inter_dists):.3f}")
    print(f"Mean Inter-Class Distance:    {np.mean(all_inter_dists):.3f}")
    print(f"Median Inter-Class Distance:  {np.median(all_inter_dists):.3f}")
    print(f"Current Match Threshold:      {DEFAULT_MATCH_THRESHOLD:.2f}")

    if np.min(all_inter_dists) <= DEFAULT_MATCH_THRESHOLD:
        colliding_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if valid_samples[i]["label"] != valid_samples[j]["label"]:
                    if dist_matrix[i, j] <= DEFAULT_MATCH_THRESHOLD:
                        colliding_pairs.append((sample_ids[i], sample_ids[j], dist_matrix[i, j]))
        print(f"\n[ALERT] {len(colliding_pairs)} inter-class pair(s) have distance <= {DEFAULT_MATCH_THRESHOLD:.2f}:")
        for s1, s2, d in colliding_pairs:
            print(f"  - {s1} <--> {s2} : MAE = {d:.3f}")
    else:
        print(f"\n[CONFIRMED] Zero inter-class collisions! Every distinct sign pair has distance > {DEFAULT_MATCH_THRESHOLD:.2f}.")

    print("\n" + "=" * 80)
    print(" VALIDATION EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
