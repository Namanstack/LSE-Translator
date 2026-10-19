"""
LSE Multimodal Model Verification & Test Suite

Executes comprehensive dummy forward-pass tests:
1. Validates tensor shapes across hand/face encoders, adaptive fusion, and classifier.
2. Checks for absence of NaN and Inf values.
3. Tests backward pass and gradient flow through all components.
4. Inspects learnable fusion weights.
"""

import os
import sys
import torch

# Ensure project root is available
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.config import ModelConfig
from models.multimodal_model import MultimodalSignModel, get_multimodal_model_summary


def test_multimodal_pipeline():
    print("=" * 70)
    print(" LSE MULTIMODAL MODEL (HANDS + FACE) ARCHITECTURE TEST")
    print("=" * 70)

    # 1. Configuration
    batch_size = 2
    sequence_length = 30
    hand_dim = 126
    face_dim = 1404
    num_classes = 5

    config = ModelConfig(
        sequence_length=sequence_length,
        hand_feature_dim=hand_dim,
        face_feature_dim=face_dim,
        hand_embedding_dim=128,
        face_embedding_dim=128,
        fusion_dim=128,
        temporal_hidden_dim=128,
        temporal_num_layers=2,
        num_classes=num_classes,
        encoder_dropout=0.2,
        fusion_dropout=0.2,
        temporal_dropout=0.3
    )

    # 2. Instantiate Model
    model = MultimodalSignModel(config=config)
    summary_txt, total_params = get_multimodal_model_summary(model)

    print(f"Total Trainable Parameters: {total_params:,}")
    print("-" * 70)

    # 3. Create Dummy Inputs
    dummy_hands = torch.randn(batch_size, sequence_length, hand_dim, dtype=torch.float32, requires_grad=True)
    dummy_face = torch.randn(batch_size, sequence_length, face_dim, dtype=torch.float32, requires_grad=True)

    print(f"Dummy Hand Input Shape:  {tuple(dummy_hands.shape)}")
    print(f"Dummy Face Input Shape:  {tuple(dummy_face.shape)}")
    print("-" * 70)

    # 4. Forward Pass
    model.train()
    output = model(dummy_hands, dummy_face)

    # 5. Output Dimension Inspections
    print("FORWARD PASS TENSOR SHAPES:")
    print(f" - Hand Embeddings:      {tuple(output.hand_embeddings.shape)} (Expected: {(batch_size, sequence_length, 128)})")
    print(f" - Face Embeddings:      {tuple(output.face_embeddings.shape)} (Expected: {(batch_size, sequence_length, 128)})")
    print(f" - Fused Features:       {tuple(output.fused_features.shape)} (Expected: {(batch_size, sequence_length, 128)})")
    print(f" - Fusion Weights:       {tuple(output.fusion_weights.shape)} (Expected: {(batch_size, sequence_length, 2)})")
    print(f" - Final Classifier Logits: {tuple(output.logits.shape)} (Expected: {(batch_size, num_classes)})")
    print("-" * 70)

    # 6. Check Sample Fusion Weights
    w_sample = output.fusion_weights[0, 0].detach().numpy()
    print(f"Sample Frame 0 Modality Weights [w_hand, w_face]: [{w_sample[0]:.4f}, {w_sample[1]:.4f}] (Sum: {w_sample.sum():.4f})")
    print("-" * 70)

    # 7. Assertions on Dimensions
    assert output.hand_embeddings.shape == (batch_size, sequence_length, 128)
    assert output.face_embeddings.shape == (batch_size, sequence_length, 128)
    assert output.fused_features.shape == (batch_size, sequence_length, 128)
    assert output.fusion_weights.shape == (batch_size, sequence_length, 2)
    assert output.logits.shape == (batch_size, num_classes)

    # 8. NaN / Inf Checks
    for name, tensor in [
        ("logits", output.logits),
        ("hand_emb", output.hand_embeddings),
        ("face_emb", output.face_embeddings),
        ("fused", output.fused_features),
        ("weights", output.fusion_weights)
    ]:
        assert not torch.isnan(tensor).any(), f"NaN found in {name}"
        assert not torch.isinf(tensor).any(), f"Inf found in {name}"

    print("[PASSED] NaN/Inf verification clean across all intermediate tensors.")

    # 9. Gradient Flow Check (Backward pass)
    loss = output.logits.sum()
    loss.backward()

    # Verify gradients reached input tensors and encoder weights
    assert dummy_hands.grad is not None and not torch.isnan(dummy_hands.grad).any()
    assert dummy_face.grad is not None and not torch.isnan(dummy_face.grad).any()
    assert model.hand_encoder.encoder[0].weight.grad is not None
    assert model.face_encoder.encoder[0].weight.grad is not None
    assert model.fusion.gating_network[0].weight.grad is not None

    print("[PASSED] Gradient flow check: Gradients successfully backpropagated through all modalities.")
    print("=" * 70)
    print("ALL ARCHITECTURE & FORWARD-PASS TESTS PASSED.")
    print("=" * 70)


if __name__ == "__main__":
    test_multimodal_pipeline()
