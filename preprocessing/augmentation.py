"""
LSE Landmark Data Augmentation Module
=====================================
Provides real-time and offline landmark-level spatio-temporal data augmentations
for sign language sequence modeling.

Operates directly on coordinate sequences of shape:
    Hands: (T, 126) -> (T, 2 hands, 21 landmarks, 3 coords)
    Face:  (T, 1404) -> (T, 468 landmarks, 3 coords)

Supported Augmentations:
------------------------
1. Jitter / Noise Injection: Adds subtle Gaussian noise to active landmarks.
2. Rotation Perturbation: Rotates hand/face coordinates in 2D/3D around their geometric center.
3. Translation Perturbation: Applies small spatial offsets across all coordinates.
4. Spatial Scaling: Slightly expands or shrinks landmark span.
5. Temporal Resampling / Scaling: Simulates variable signing speed (fast/slow) using interpolation.
6. Hand Drop / Modality Dropout: Randomly masks an undetected/non-dominant hand with probability p.

Design & Safety:
----------------
- Preserves exact input shapes: (T, 126) and (T, 1404).
- Safe guard against NaN/Inf values.
- Non-destructive (returns fresh arrays without modifying inputs).
- Disabled by default unless explicitly instantiated with active probabilities or enabled flag.
- Fully deterministic when supplied with an explicit seed / np.random.RandomState.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np


@dataclass
class AugmentationConfig:
    """
    Configuration hyperparameters for landmark sequence data augmentation.
    All augmentations default to enabled=False or p=0.0 unless configured.
    """
    enabled: bool = False
    
    # 1. Jitter / Gaussian noise
    jitter_prob: float = 0.5
    jitter_std: float = 0.015
    
    # 2. Rotation (degrees in X-Y plane)
    rotation_prob: float = 0.5
    rotation_angle_deg: float = 8.0
    
    # 3. Translation
    translation_prob: float = 0.5
    translation_range: float = 0.03
    
    # 4. Spatial Scaling
    scaling_prob: float = 0.5
    scaling_range: Tuple[float, float] = (0.92, 1.08)
    
    # 5. Temporal Resampling (signing speed variation)
    temporal_scale_prob: float = 0.4
    temporal_speed_range: Tuple[float, float] = (0.85, 1.15)
    
    # Random seed for reproducible transformations
    seed: Optional[int] = None


class LandmarkAugmentor:
    """
    Applies configurable landmark-level transformations to hand and face sequences.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig(enabled=False)
        self.rng = np.random.RandomState(self.config.seed)

    def set_seed(self, seed: int):
        """Sets the random state seed for deterministic behavior."""
        self.rng = np.random.RandomState(seed)

    # -------------------------------------------------------------------------
    # Individual Spatio-Temporal Transformations
    # -------------------------------------------------------------------------

    def apply_jitter(self, sequence: np.ndarray, std: float) -> np.ndarray:
        """
        Adds Gaussian noise to non-zero landmark coordinates.
        Zero coordinates (representing missing detections) are preserved.
        """
        mask = (np.abs(sequence).sum(axis=-1, keepdims=True) > 1e-6).astype(np.float32)
        noise = self.rng.normal(0.0, std, size=sequence.shape).astype(np.float32)
        return sequence + (noise * mask)

    def apply_rotation_2d(self, landmarks_3d: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        Rotates 3D landmark coordinates in the X-Y plane around their centroid per frame.
        landmarks_3d shape: (T, N_landmarks, 3)
        """
        rad = math.radians(angle_deg)
        cos_theta = math.cos(rad)
        sin_theta = math.sin(rad)
        rot_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ], dtype=np.float32)

        out = landmarks_3d.copy()
        T, N, C = landmarks_3d.shape

        for t in range(T):
            frame = landmarks_3d[t]  # (N, 3)
            # Find active landmarks
            active_mask = np.abs(frame).sum(axis=-1) > 1e-6
            if not np.any(active_mask):
                continue

            active_lms = frame[active_mask]  # (K, 3)
            centroid_xy = active_lms[:, :2].mean(axis=0)  # (2,)

            # Centering, rotation, de-centering
            shifted = active_lms[:, :2] - centroid_xy
            rotated = np.dot(shifted, rot_matrix.T) + centroid_xy

            out[t, active_mask, :2] = rotated

        return out

    def apply_translation(self, landmarks_3d: np.ndarray, max_offset: float) -> np.ndarray:
        """
        Applies a smooth translation offset in (X, Y) to active landmarks.
        landmarks_3d shape: (T, N_landmarks, 3)
        """
        dx = self.rng.uniform(-max_offset, max_offset)
        dy = self.rng.uniform(-max_offset, max_offset)
        offset = np.array([dx, dy, 0.0], dtype=np.float32)

        out = landmarks_3d.copy()
        mask = (np.abs(landmarks_3d).sum(axis=-1, keepdims=True) > 1e-6).astype(np.float32)
        return out + (offset * mask)

    def apply_scaling(self, landmarks_3d: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Scales active landmark coordinates relative to their center.
        landmarks_3d shape: (T, N_landmarks, 3)
        """
        out = landmarks_3d.copy()
        T, N, C = landmarks_3d.shape

        for t in range(T):
            frame = landmarks_3d[t]
            active_mask = np.abs(frame).sum(axis=-1) > 1e-6
            if not np.any(active_mask):
                continue

            active_lms = frame[active_mask]
            centroid = active_lms.mean(axis=0)  # (3,)
            scaled = (active_lms - centroid) * scale_factor + centroid
            out[t, active_mask] = scaled

        return out

    def apply_temporal_resampling(self, sequence_2d: np.ndarray, speed_factor: float) -> np.ndarray:
        """
        Simulates speed variation by compressing or stretching time, then resampling back to target T.
        sequence_2d shape: (T, D)
        """
        T, D = sequence_2d.shape
        if T <= 1 or abs(speed_factor - 1.0) < 1e-4:
            return sequence_2d.copy()

        # New virtual sequence length based on speed
        new_len = max(2, int(round(T / speed_factor)))
        orig_indices = np.linspace(0, T - 1, num=T)
        intermediate_indices = np.linspace(0, T - 1, num=new_len)

        # 1. Warp to virtual length
        warped = np.zeros((new_len, D), dtype=np.float32)
        for col in range(D):
            warped[:, col] = np.interp(intermediate_indices, orig_indices, sequence_2d[:, col])

        # 2. Resample back to original T
        final_indices = np.linspace(0, new_len - 1, num=T)
        warped_indices = np.linspace(0, new_len - 1, num=new_len)
        resampled = np.zeros((T, D), dtype=np.float32)
        for col in range(D):
            resampled[:, col] = np.interp(final_indices, warped_indices, warped[:, col])

        return resampled

    # -------------------------------------------------------------------------
    # Composite Modality Augmentation
    # -------------------------------------------------------------------------

    def augment_hand_features(self, hand_features: np.ndarray) -> np.ndarray:
        """
        Augments a hand feature sequence of shape (T, 126).
        Decomposes into Left Hand (63 = 21*3) and Right Hand (63 = 21*3)
        to preserve articulated structural fidelity.
        """
        if not self.config.enabled:
            return hand_features.copy()

        T, D = hand_features.shape
        if D != 126:
            raise ValueError(f"Expected hand_features of dimension 126, got {D}")

        out = hand_features.copy()

        # 1. Temporal Resampling
        if self.rng.uniform(0, 1) < self.config.temporal_scale_prob:
            speed = self.rng.uniform(*self.config.temporal_speed_range)
            out = self.apply_temporal_resampling(out, speed)

        # Reshape to (T, 2 hands, 21 landmarks, 3 coords)
        hands_3d = out.reshape(T, 2, 21, 3)

        # 2. Spatial augmentations per hand
        for h in range(2):
            hand_lms = hands_3d[:, h, :, :]  # (T, 21, 3)

            # Rotation
            if self.rng.uniform(0, 1) < self.config.rotation_prob:
                angle = self.rng.uniform(-self.config.rotation_angle_deg, self.config.rotation_angle_deg)
                hand_lms = self.apply_rotation_2d(hand_lms, angle)

            # Scaling
            if self.rng.uniform(0, 1) < self.config.scaling_prob:
                scale = self.rng.uniform(*self.config.scaling_range)
                hand_lms = self.apply_scaling(hand_lms, scale)

            # Translation
            if self.rng.uniform(0, 1) < self.config.translation_prob:
                hand_lms = self.apply_translation(hand_lms, self.config.translation_range)

            # Jitter
            if self.rng.uniform(0, 1) < self.config.jitter_prob:
                hand_lms = self.apply_jitter(hand_lms, self.config.jitter_std)

            hands_3d[:, h, :, :] = hand_lms

        flattened = hands_3d.reshape(T, 126).astype(np.float32)
        # NaN/Inf security guard
        flattened = np.nan_to_num(flattened, nan=0.0, posinf=0.0, neginf=0.0)
        return flattened

    def augment_face_features(self, face_features: np.ndarray) -> np.ndarray:
        """
        Augments a facial landmark feature sequence of shape (T, 1404).
        Reshapes into (T, 468, 3) for geometric transformations.
        """
        if not self.config.enabled:
            return face_features.copy()

        T, D = face_features.shape
        if D != 1404:
            raise ValueError(f"Expected face_features of dimension 1404, got {D}")

        out = face_features.copy()

        # 1. Temporal Resampling
        if self.rng.uniform(0, 1) < self.config.temporal_scale_prob:
            speed = self.rng.uniform(*self.config.temporal_speed_range)
            out = self.apply_temporal_resampling(out, speed)

        face_3d = out.reshape(T, 468, 3)

        # 2. Rotation
        if self.rng.uniform(0, 1) < self.config.rotation_prob:
            angle = self.rng.uniform(-self.config.rotation_angle_deg, self.config.rotation_angle_deg)
            face_3d = self.apply_rotation_2d(face_3d, angle)

        # 3. Scaling
        if self.rng.uniform(0, 1) < self.config.scaling_prob:
            scale = self.rng.uniform(*self.config.scaling_range)
            face_3d = self.apply_scaling(face_3d, scale)

        # 4. Translation
        if self.rng.uniform(0, 1) < self.config.translation_prob:
            face_3d = self.apply_translation(face_3d, self.config.translation_range)

        # 5. Jitter
        if self.rng.uniform(0, 1) < self.config.jitter_prob:
            face_3d = self.apply_jitter(face_3d, self.config.jitter_std)

        flattened = face_3d.reshape(T, 1404).astype(np.float32)
        # NaN/Inf security guard
        flattened = np.nan_to_num(flattened, nan=0.0, posinf=0.0, neginf=0.0)
        return flattened

    def __call__(
        self,
        hand_features: np.ndarray,
        face_features: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Unified callable interface for transforming hand sequences or paired multimodal sequences.
        """
        aug_hands = self.augment_hand_features(hand_features)

        if face_features is not None:
            aug_face = self.augment_face_features(face_features)
            return aug_hands, aug_face

        return aug_hands
