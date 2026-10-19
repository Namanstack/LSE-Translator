# LSE-Translator: Multimodal Spanish Sign Language Recognition & Translation

An end-to-end multimodal machine learning system for real-time **Spanish Sign Language (Lengua de Signos Española - LSE)** recognition and translation.

The architecture processes synchronized spatio-temporal streams of **articulated hand gestures** (MediaPipe Hands) and **facial non-manual expressions** (MediaPipe Face Mesh), combining them through adaptive modality fusion and recurrent temporal modeling (LSTM).

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [System Architecture](#system-architecture)
3. [Tensor Shapes & Data Contracts](#tensor-shapes--data-contracts)
4. [Project Directory Structure](#project-directory-structure)
5. [Installation & Setup](#installation--setup)
6. [Pipeline Workflows & Usage](#pipeline-workflows--usage)
   - [Live Tracking (Hands & Face)](#1-live-tracking-hands--face)
   - [Dataset Collection & Recording](#2-dataset-collection--recording)
   - [Dataset Validation](#3-dataset-validation)
   - [Landmark Normalization & Preprocessing](#4-landmark-normalization--preprocessing)
   - [Video Corpus Ingestion Utility](#5-video-corpus-ingestion-utility)
   - [Data Augmentation](#6-landmark-level-data-augmentation)
   - [Real-Time Inference Pipeline](#7-real-time-inference-pipeline)
7. [Signer-Independent Split Strategy](#signer-independent-split-strategy)
8. [Training Framework](#training-framework)
9. [Verification & Testing](#verification--testing)
10. [Current Limitations & Real Dataset Status](#current-limitations--real-dataset-status)

---

## Problem Statement

Sign language is a rich, natural, multidimensional language consisting of:
- **Manual features**: Hand shapes, orientations, trajectories, and locations.
- **Non-manual features**: Facial expressions, head tilts, eyebrow positions, and mouthings that provide critical grammatical and contextual disambiguation.

LSE-Translator builds a standardized, robust, and translation-invariant pipeline that processes multimodal visual cues into text glosses and full Spanish/English translations.

---

## System Architecture

```
                       +-----------------------------+
                       |  Live Video / Video Corpus  |
                       +--------------+--------------+
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
        +--------------------------+     +--------------------------+
        |     MediaPipe Hands      |     |   MediaPipe Face Mesh    |
        | 2 Hands x 21 Landmarks   |     |      468 Landmarks       |
        +-------------+------------+     +-------------+------------+
                      |                                |
                      v                                v
        +--------------------------+     +--------------------------+
        | Wrist-Centered Geometry  |     |   Nose-Centric Norm &    |
        |  & Max-Distance Scale    |     |   Distance Invariance    |
        +-------------+------------+     +-------------+------------+
                      | (T, 126)                       | (T, 1404)
                      +---------------+----------------+
                                      |
                      v               v                v
        +--------------------+----------------+--------------------+
        | HandFeatureEncoder |                | FaceFeatureEncoder |
        | Linear -> 128-d    |                | Linear -> 128-d    |
        +---------+----------+                +---------+----------+
                  |                                     |
                  +-----------------+-------------------+
                                    |
                                    v
                     +-----------------------------+
                     |  Adaptive Gating & Fusion   |
                     | Softmax Modality Weighting  |
                     |  w_hand * E_h + w_face * E_f|
                     +--------------+--------------+
                                    | (B, T, 128)
                                    v
                     +-----------------------------+
                     |  Temporal Classifier (LSTM) |
                     | 2 Layers, Hidden Dim = 128  |
                     +--------------+--------------+
                                    |
                                    v
                     +-----------------------------+
                     |    Linear Classification    |
                     |      Logits -> Softmax      |
                     +--------------+--------------+
                                    |
                                    v
                     +-----------------------------+
                     |    LSE Translation Layer    |
                     | Gloss -> Spanish / English  |
                     +-----------------------------+
```

---

## Tensor Shapes & Data Contracts

| Modality / Tensor | Raw Dimension | Normalized Dimension | Sequence Shape $(B, T, D)$ |
| :--- | :--- | :--- | :--- |
| **Hand Landmarks** | $2 \text{ hands} \times 21 \text{ landmarks} \times 3\ (x, y, z)$ | $126$ coordinates | `(B, 30, 126)` |
| **Face Landmarks** | $468 \text{ landmarks} \times 3\ (x, y, z)$ | $1404$ coordinates | `(B, 30, 1404)` |
| **Fused Embedding** | Concatenated ($128 + 128$) | Gated projection ($128$) | `(B, 30, 128)` |
| **LSTM Hidden Output** | $128$ hidden units | Final step / pool | `(B, 128)` |
| **Class Prediction** | $N$ vocabulary classes | Logits scalar per class | `(B, num_classes)` |

---

## Project Directory Structure

```
E:\LSE_TRANSLATOR\
├── data/
│   ├── labels/                   # Vocabulary definitions and index mappings
│   ├── raw/                      # Raw recording CSVs (data/raw/<LABEL>/<signer>_<id>.csv)
│   └── processed/                # Standardized .npz files (data/processed/<LABEL>/<id>.npz)
├── models/
│   ├── config.py                 # Central architectural & vocabulary configurations
│   ├── multimodal_model.py       # Unified MultimodalSignModel implementation
│   ├── temporal_classifier.py    # Multi-layer temporal LSTM classifier
│   ├── hands/                    # Hand encoder, PyTorch dataset, and hands-only baseline
│   ├── face/                     # Facial feature encoder and dimensionality reduction
│   ├── fusion/                   # Adaptive gating fusion layer
│   └── checkpoints/              # Checkpoint storage (ignored in Git)
├── preprocessing/
│   ├── webcam_hands.py           # Real-time hand landmark tracker & preview
│   ├── webcam_face.py            # Real-time face mesh tracker & preview
│   ├── collect_hand_landmarks.py # Hand gesture sequence collector
│   ├── collect_face_landmarks.py # Facial landmark sequence collector
│   ├── normalize_landmarks.py    # Wrist-centering and scale-invariance transformation
│   ├── validate_dataset.py       # Hand dataset schema, frame-count, and NaN/Inf validation
│   ├── validate_face_dataset.py  # Face dataset validation and bounding check
│   ├── convert_video_corpus.py   # Batch video dataset ingestion & standardization utility
│   └── augmentation.py           # Landmark-level spatio-temporal data augmentation module
├── recognition/
│   └── realtime_inference.py     # Real-time rolling buffer inference HUD
├── translation/
│   └── translator.py             # LSE token-to-gloss & bilingual translator (ES / EN)
├── training/
│   ├── train_config.py           # Hyperparameters and path configuration dataclass
│   ├── multimodal_dataset.py     # Multimodal PyTorch dataset loader with missing-modality fallback
│   ├── signer_split.py           # Leakage-free signer-independent train/val/test splitting
│   ├── trainer.py                # PyTorch training engine with AdamW, EarlyStopping, & Checkpointing
│   └── train.py                  # CLI entry point for model training
├── results/
│   └── evaluate.py               # Evaluation engine (Accuracy, F1, latency, confusion matrices)
├── tests/                        # Comprehensive automated test suite
├── requirements.txt              # Core project dependencies
├── PROJECT_STATUS.md             # Milestone tracking and development handoff report
└── README.md                     # System documentation
```

---

## Installation & Setup

### 1. Environment Requirements
- Python 3.10 to 3.12 (Tested on Python 3.12.4 on Windows)
- PyTorch (CPU or CUDA enabled)
- MediaPipe
- OpenCV

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Pipeline Workflows & Usage

### 1. Live Tracking (Hands & Face)
Preview live camera feeds with MediaPipe landmark overlays:
```bash
# Hands tracking preview
python preprocessing/webcam_hands.py

# Facial mesh tracking preview
python preprocessing/webcam_face.py
```

### 2. Dataset Collection & Recording
Capture synchronized $T=30$ frame sequences directly from your webcam:
```bash
# Record hand sequence (Press SPACE to start recording, R to reset, Q to quit)
python preprocessing/collect_hand_landmarks.py --label HOLA --signer signer01 --recording 1

# Record face sequence
python preprocessing/collect_face_landmarks.py --label HOLA --signer signer01 --recording_id 1
```

### 3. Dataset Validation
Run structural and numeric integrity checks across all raw and processed recordings:
```bash
# Validate hand dataset
python preprocessing/validate_dataset.py --data_dir data/raw

# Validate face dataset
python preprocessing/validate_face_dataset.py --data_dir data/raw
```

### 4. Landmark Normalization & Preprocessing
Transforms raw CSV recordings into translation-invariant, scale-invariant `.npz` arrays:
```bash
python preprocessing/normalize_landmarks.py --raw_dir data/raw --processed_dir data/processed
```

### 5. Video Corpus Ingestion Utility
Convert external video datasets (e.g. MP4, AVI, MOV) into standardized `.npz` files:
```bash
python preprocessing/convert_video_corpus.py --input_dir path/to/raw_videos --output_dir data/processed
```

### 6. Landmark-Level Data Augmentation
Dynamic augmentation can be enabled in datasets during training or used offline:
```python
from preprocessing.augmentation import AugmentationConfig, LandmarkAugmentor

# Configure desired transformation strengths
config = AugmentationConfig(
    enabled=True,
    jitter_prob=0.5,
    jitter_std=0.015,
    rotation_prob=0.5,
    rotation_angle_deg=8.0,
    translation_prob=0.5,
    translation_range=0.03,
    scaling_prob=0.5,
    scaling_range=(0.92, 1.08),
    temporal_scale_prob=0.4,
    temporal_speed_range=(0.85, 1.15)
)
augmentor = LandmarkAugmentor(config)

# Apply non-destructively to hand & face sequences
aug_hands, aug_face = augmentor(hand_seq_30x126, face_seq_30x1404)
```

### 7. Real-Time Inference Pipeline
Runs the live webcam HUD with rolling sliding window buffer ($T=30$) and translation output:
```bash
python recognition/realtime_inference.py --checkpoint models/checkpoints/best_model.pt
```
*Note: When no checkpoint exists, the system runs in safe Demo/Verification mode (`MODEL STATUS: NOT TRAINED - INFERENCE DISABLED`).*

### 8. Live Web Demonstration (Gradio UI)
Launch the browser-based live demonstration for faculty review:
```bash
python app/web_demo.py
```
*Provides a unified interface displaying simultaneous hand + face landmark tracking, real-time FPS/telemetry, multimodal alignment indicators, and architecture flow.*


---

## Signer-Independent Split Strategy

To guarantee true model generalization and prevent overfitting to individual signer morphology:
1. All unique signers in `data/processed/` are discovered.
2. Signers are partitioned strictly into **Train**, **Validation**, and **Test** sets.
3. No signer present in the training set ever appears in validation or test splits.
4. If signers are unknown, splitting defaults to stratified sample partitioning with explicit leakage warnings.

```bash
# Generate signer-independent splits
python training/signer_split.py --data_dir data/processed --val_ratio 0.2 --test_ratio 0.1
```

---

## Training Framework

The training pipeline ([`training/trainer.py`](file:///e:/LSE_TRANSLATOR/training/trainer.py)) is implemented with:
- **Optimizer**: AdamW with weight decay.
- **LR Scheduling**: Cosine Annealing or StepLR.
- **Loss**: Multi-class Cross-Entropy with optional label smoothing.
- **Early Stopping**: Configurable patience on validation loss.
- **Safety**: Automated NaN/Inf gradient detection and anomaly interception.
- **Checkpointing**: Automatic saving of `best_model.pt` and periodic snapshots.

```bash
# Run training once real multi-signer dataset is acquired
python training/train.py --data_dir data/processed --epochs 50 --batch_size 16 --lr 1e-3
```

---

## Verification & Testing

Execute the complete automated test suite (24 tests covering models, datasets, transforms, real-time buffers, and training loops):
```bash
python -m pytest -v tests
```

---

## Current Limitations & Real Dataset Status

> [!IMPORTANT]
> **Real Model Training Has Not Been Performed Yet.**
> The current local dataset files in `data/raw/` and `data/processed/` consist only of minimal verification samples used to test pipelines and integration.
>
> In accordance with scientific integrity standards:
> - No fake accuracy, loss, or F1 scores are claimed.
> - No confusion matrices have been generated from placeholder data.
> - Real model training and evaluation will be performed once an adequate multi-signer, multi-class LSE dataset is acquired.