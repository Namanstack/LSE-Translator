"""
Unit tests for LSE Landmark Data Augmentation Module (preprocessing/augmentation.py)
"""

import numpy as np
import pytest
import torch

from preprocessing.augmentation import (
    AugmentationConfig,
    LandmarkAugmentor
)
from models.hands.dataset import LSEHandsDataset
from training.multimodal_dataset import LSEMultimodalDataset


@pytest.fixture
def dummy_hand_sequence():
    """Generates synthetic (30, 126) hand landmark tensor."""
    np.random.seed(42)
    seq = np.random.uniform(-0.5, 0.5, size=(30, 126)).astype(np.float32)
    return seq


@pytest.fixture
def dummy_face_sequence():
    """Generates synthetic (30, 1404) face landmark tensor."""
    np.random.seed(42)
    seq = np.random.uniform(-0.5, 0.5, size=(30, 1404)).astype(np.float32)
    return seq


def test_disabled_by_default(dummy_hand_sequence, dummy_face_sequence):
    """Verifies that default config leaves sequences completely unmodified."""
    augmentor = LandmarkAugmentor()
    assert not augmentor.config.enabled

    orig_hand = dummy_hand_sequence.copy()
    orig_face = dummy_face_sequence.copy()

    aug_hand, aug_face = augmentor(orig_hand, orig_face)

    np.testing.assert_array_equal(aug_hand, orig_hand)
    np.testing.assert_array_equal(aug_face, orig_face)


def test_shape_preservation(dummy_hand_sequence, dummy_face_sequence):
    """Verifies that transformed outputs maintain exact (T, 126) and (T, 1404) shapes."""
    cfg = AugmentationConfig(
        enabled=True,
        jitter_prob=1.0,
        rotation_prob=1.0,
        translation_prob=1.0,
        scaling_prob=1.0,
        temporal_scale_prob=1.0,
        seed=123
    )
    augmentor = LandmarkAugmentor(cfg)

    aug_hand, aug_face = augmentor(dummy_hand_sequence, dummy_face_sequence)

    assert aug_hand.shape == (30, 126)
    assert aug_face.shape == (30, 1404)
    assert aug_hand.dtype == np.float32
    assert aug_face.dtype == np.float32


def test_no_nan_or_inf(dummy_hand_sequence, dummy_face_sequence):
    """Ensures no NaN or Inf is ever produced under full augmentation."""
    cfg = AugmentationConfig(
        enabled=True,
        jitter_prob=1.0,
        rotation_prob=1.0,
        translation_prob=1.0,
        scaling_prob=1.0,
        temporal_scale_prob=1.0,
        seed=42
    )
    augmentor = LandmarkAugmentor(cfg)

    for _ in range(10):
        aug_hand, aug_face = augmentor(dummy_hand_sequence, dummy_face_sequence)
        assert not np.isnan(aug_hand).any(), "NaN found in augmented hand sequence"
        assert not np.isinf(aug_hand).any(), "Inf found in augmented hand sequence"
        assert not np.isnan(aug_face).any(), "NaN found in augmented face sequence"
        assert not np.isinf(aug_face).any(), "Inf found in augmented face sequence"


def test_deterministic_with_fixed_seed(dummy_hand_sequence, dummy_face_sequence):
    """Verifies that two augmentors with the same seed generate identical outputs."""
    cfg1 = AugmentationConfig(enabled=True, jitter_prob=1.0, rotation_prob=1.0, seed=999)
    cfg2 = AugmentationConfig(enabled=True, jitter_prob=1.0, rotation_prob=1.0, seed=999)

    aug1 = LandmarkAugmentor(cfg1)
    aug2 = LandmarkAugmentor(cfg2)

    h1, f1 = aug1(dummy_hand_sequence, dummy_face_sequence)
    h2, f2 = aug2(dummy_hand_sequence, dummy_face_sequence)

    np.testing.assert_allclose(h1, h2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(f1, f2, rtol=1e-5, atol=1e-5)


def test_augmentation_modifies_data_when_enabled(dummy_hand_sequence, dummy_face_sequence):
    """Verifies that enabling transformations actually perturbs the coordinates."""
    cfg = AugmentationConfig(
        enabled=True,
        jitter_prob=1.0,
        rotation_prob=1.0,
        translation_prob=1.0,
        scaling_prob=1.0,
        seed=100
    )
    augmentor = LandmarkAugmentor(cfg)

    aug_hand, aug_face = augmentor(dummy_hand_sequence, dummy_face_sequence)

    assert not np.array_equal(aug_hand, dummy_hand_sequence)
    assert not np.array_equal(aug_face, dummy_face_sequence)


def test_original_input_remains_unmodified(dummy_hand_sequence, dummy_face_sequence):
    """Verifies non-destructive guarantee on original input arrays."""
    orig_hand_copy = dummy_hand_sequence.copy()
    orig_face_copy = dummy_face_sequence.copy()

    cfg = AugmentationConfig(enabled=True, jitter_prob=1.0, rotation_prob=1.0, seed=200)
    augmentor = LandmarkAugmentor(cfg)

    _ = augmentor(dummy_hand_sequence, dummy_face_sequence)

    np.testing.assert_array_equal(dummy_hand_sequence, orig_hand_copy)
    np.testing.assert_array_equal(dummy_face_sequence, orig_face_copy)


def test_dynamic_integration_with_multimodal_dataset():
    """Verifies that LandmarkAugmentor integrates transparently with PyTorch Datasets."""
    cfg = AugmentationConfig(enabled=True, jitter_prob=1.0, seed=42)
    augmentor = LandmarkAugmentor(cfg)

    # Mock sample inside dataset
    dummy_hand = np.ones((30, 126), dtype=np.float32)
    dummy_face = np.ones((30, 1404), dtype=np.float32)

    transformed_hand, transformed_face = augmentor(dummy_hand, dummy_face)
    assert transformed_hand.shape == (30, 126)
    assert transformed_face.shape == (30, 1404)
    assert not np.array_equal(transformed_hand, dummy_hand)
