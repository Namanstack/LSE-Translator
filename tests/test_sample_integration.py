"""
LSE Real Sample Integration Test (Modality Integration Only)

INTEGRATION TEST ONLY — NOT TRAINING/EVALUATION.
Verifies that real processed landmark recordings for hands and face can be loaded,
aligned, and passed forward through the MultimodalSignModel end-to-end.
"""

import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.config import ModelConfig
from models.multimodal_model import MultimodalSignModel
from translation.translator import LSETranslator


def run_sample_integration():
    print("=" * 70)
    print(" INTEGRATION TEST ONLY — NOT TRAINING/EVALUATION")
    print("=" * 70)

    # 1. Paths to existing processed recordings
    hand_file = os.path.join("data", "processed", "HOLA", "signer01_001.npz")
    face_file = os.path.join("data", "processed", "HOLA", "face_signer01_001.npz")

    if not os.path.exists(hand_file) or not os.path.exists(face_file):
        print(f"[Error] Required test recordings missing:\n Hand: {hand_file}\n Face: {face_file}", file=sys.stderr)
        return

    # 2. Load Processed Samples
    hand_data = np.load(hand_file, allow_pickle=True)
    face_data = np.load(face_file, allow_pickle=True)

    hand_feat = hand_data["features"]  # (30, 126)
    face_feat = face_data["features"]  # (30, 1404)
    label = str(hand_data["label"])

    print(f" Loaded Hand Recording: {hand_file} | Shape: {hand_feat.shape}")
    print(f" Loaded Face Recording: {face_file} | Shape: {face_feat.shape}")
    print(f" Ground-Truth Label:    {label}")
    print("-" * 70)

    # Convert to batch tensors: (1, 30, 126) and (1, 30, 1404)
    hand_tensor = torch.tensor(hand_feat, dtype=torch.float32).unsqueeze(0)
    face_tensor = torch.tensor(face_feat, dtype=torch.float32).unsqueeze(0)

    # 3. Instantiate Multimodal Architecture
    config = ModelConfig(
        sequence_length=30,
        hand_feature_dim=126,
        face_feature_dim=1404,
        hand_embedding_dim=128,
        face_embedding_dim=128,
        fusion_dim=128,
        temporal_hidden_dim=128,
        num_classes=5
    )
    model = MultimodalSignModel(config=config)
    model.eval()

    # 4. Perform Forward Pass (NO training, NO optimizer)
    with torch.no_grad():
        out = model(hand_tensor, face_tensor)

    print(" MULTIMODAL FORWARD PASS RESULTS:")
    print(f"  - Hand Emb Shape:   {tuple(out.hand_embeddings.shape)}")
    print(f"  - Face Emb Shape:   {tuple(out.face_embeddings.shape)}")
    print(f"  - Fused Feat Shape: {tuple(out.fused_features.shape)}")
    print(f"  - Fusion Wts Shape: {tuple(out.fusion_weights.shape)}")
    print(f"  - Logits Shape:     {tuple(out.logits.shape)}")

    # 5. Translation Layer Test
    translator = LSETranslator()
    pred_class_idx = int(torch.argmax(out.logits, dim=1).item())
    pred_label = config.target_vocab[pred_class_idx]
    translated_es = translator.translate_label(pred_label, target_lang="es")
    translated_en = translator.translate_label(pred_label, target_lang="en")

    print("-" * 70)
    print(f" Translation Layer Simulation:")
    print(f"  - Simulated Predicted Class: {pred_label} (Index {pred_class_idx})")
    print(f"  - Spanish Text:              '{translated_es}'")
    print(f"  - English Text:              '{translated_en}'")
    print("-" * 70)
    print("[PASSED] Sample Integration Test completed cleanly.")
    print("=" * 70)


if __name__ == "__main__":
    run_sample_integration()
