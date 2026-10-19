"""
Tests for preprocessing/convert_video_corpus.py
"""

import os
import tempfile
import numpy as np
import cv2
import pytest

from preprocessing.convert_video_corpus import (
    parse_video_metadata,
    resample_sequence,
    convert_single_video,
    discover_video_files,
    convert_corpus
)
from models.hands.dataset import LSEHandsDataset
from training.multimodal_dataset import LSEMultimodalDataset


def create_dummy_video(filepath: str, num_frames: int = 45, width: int = 128, height: int = 128):
    """Generates a dummy synthetic video for ingestion testing."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filepath, fourcc, 30.0, (width, height))

    for i in range(num_frames):
        # Draw a synthetic moving circle so frames have varying content
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + (i % 80), 20 + (i % 80)), 10, (255, 255, 255), -1)
        out.write(frame)

    out.release()


def test_metadata_parsing_hierarchical():
    video_path = os.path.join("corpus_root", "HOLA", "signer02_rec01.mp4")
    meta = parse_video_metadata(video_path, root_input_dir="corpus_root")
    assert meta["label"] == "HOLA"
    assert meta["signer"] == "signer02"
    assert "rec01" in meta["recording_id"]


def test_metadata_parsing_unknown_signer():
    video_path = os.path.join("corpus_root", "GRACIAS", "clip_0042.avi")
    meta = parse_video_metadata(video_path, root_input_dir="corpus_root")
    assert meta["label"] == "GRACIAS"
    assert meta["signer"] == "unknown"  # Clearly preserves limitation without inventing IDs


def test_resample_sequence():
    # Variable length sequence of 50 frames, 126 features
    orig = np.random.randn(50, 126).astype(np.float32)
    resampled = resample_sequence(orig, target_length=30)
    assert resampled.shape == (30, 126)
    assert not np.isnan(resampled).any()


def test_end_to_end_corpus_ingestion_and_dataset_compatibility():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "raw_videos")
        output_dir = os.path.join(tmpdir, "processed")

        # Create two dummy videos under label folders
        v1_path = os.path.join(input_dir, "HOLA", "signer01_trial01.mp4")
        v2_path = os.path.join(input_dir, "ADIOS", "signer02_trial01.mp4")
        create_dummy_video(v1_path, num_frames=20)
        create_dummy_video(v2_path, num_frames=40)

        # Run corpus ingestion
        summary = convert_corpus(
            input_dir=input_dir,
            output_dir=output_dir,
            extract_face=True,
            target_seq_len=30
        )

        assert summary["total"] == 2
        assert summary["succeeded"] == 2
        assert summary["failed"] == 0

        # Verify produced files are directly loadable by LSEHandsDataset
        hands_ds = LSEHandsDataset(data_dir=output_dir)
        assert len(hands_ds) == 2
        x_h, y_h, meta_h = hands_ds[0]
        assert x_h.shape == (30, 126)
        assert not torch_has_nan(x_h)

        # Verify produced files are directly loadable by LSEMultimodalDataset
        multi_ds = LSEMultimodalDataset(data_dir=output_dir, require_face=False)
        assert len(multi_ds) == 2
        hand_t, face_t, label_t, meta_m = multi_ds[0]
        assert hand_t.shape == (30, 126)
        assert face_t.shape == (30, 1404)
        assert not torch_has_nan(hand_t)
        assert not torch_has_nan(face_t)


def torch_has_nan(tensor):
    import torch
    return bool(torch.isnan(tensor).any() or torch.isinf(tensor).any())
