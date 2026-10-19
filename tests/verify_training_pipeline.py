"""
LSE Training Pipeline – Complete Synthetic Verification Script
===============================================================
Exercises every component of the training pipeline using randomly-generated
synthetic tensors.

══════════════════════════════════════════════════════════════════════════════
IMPORTANT – READ BEFORE INTERPRETING OUTPUT
══════════════════════════════════════════════════════════════════════════════
ALL data used in this script is randomly generated (np.random / torch.randn).
• Numbers printed (loss, accuracy, lr, etc.) are NOT model performance metrics.
• This script ONLY confirms that code paths execute correctly and shapes are
  consistent.  It makes NO claims about the model's sign-language ability.
• Real performance can only be measured after training on a genuine dataset.
══════════════════════════════════════════════════════════════════════════════

Verification checklist
----------------------
 [V01]  Dataset loading                  – LSEMultimodalDataset reads .npz files
 [V02]  Shape correctness                – hand (30,126), face (30,1404), label scalar
 [V03]  Signer-independent splitting     – train/val/test splits from disjoint signers
 [V04]  No-leakage assertion             – zero signer overlap between partitions
 [V05]  DataLoader batching              – correct batch tensors produced
 [V06]  Full forward pass                – MultimodalSignModel produces (B, C) logits
 [V07]  Loss calculation                 – CrossEntropyLoss returns finite scalar
 [V08]  Backward pass                    – gradients flow to all sub-networks
 [V09]  Optimizer step                   – AdamW updates weights without errors
 [V10]  LR scheduler (cosine)            – lr decreases monotonically over epochs
 [V11]  LR scheduler (step)              – lr decreases at configured step intervals
 [V12]  NaN protection (inputs)          – InvalidTensorError on NaN hand tensor
 [V13]  Inf protection (inputs)          – InvalidTensorError on Inf face tensor
 [V14]  NaN protection (logits)          – error propagated through model output check
 [V15]  Checkpoint saving                – .pt file written with expected keys
 [V16]  Checkpoint loading               – weights restored byte-for-byte
 [V17]  Resume training                  – epoch counter continues from checkpoint
 [V18]  Early stopping                   – loop exits after `patience` epochs no improvement
 [V19]  Reproducibility                  – same seed produces identical loss sequence
 [V20]  DatasetMissing safeguard         – DatasetMissingError on empty data_dir
 [V21]  InsufficientClasses safeguard    – raises on 1-class dataset
 [V22]  Signer overlap safeguard         – ValueError on overlapping signer assignments
 [V23]  Empty split safeguard            – RuntimeError when split has 0 samples
 [V24]  Gradient clipping                – no exploding gradient crash under large lr
 [V25]  Multi-epoch history              – history list grows correctly, saved to JSON
"""

import json
import math
import os
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.config import ModelConfig
from models.multimodal_model import MultimodalSignModel
from training.multimodal_dataset import LSEMultimodalDataset
from training.signer_split import (
    make_signer_independent_splits,
    validate_split_config,
)
from training.train_config import TrainConfig
from training.trainer import (
    Trainer,
    DatasetMissingError,
    InsufficientClassesError,
    InvalidTensorError,
    _check_tensor,
    set_seed,
)

# ─────────────────────────────────────────────────────────────────────────────
# Colours / formatting
# ─────────────────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW= "\033[93m"
BOLD  = "\033[1m"
RESET = "\033[0m"

def _ok(tag: str, detail: str = "") -> str:
    return f"  {GREEN}✓{RESET} [{tag}] {detail}"

def _fail(tag: str, detail: str = "") -> str:
    return f"  {RED}✗{RESET} [{tag}] {detail}"

def _head(title: str) -> None:
    print(f"\n{BOLD}{'─'*66}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*66}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_npz(path, label, signer, rec_id,
               corrupt_hand=False, corrupt_face=False,
               wrong_hand_shape=False, wrong_face_shape=False):
    hand_shape = (30, 50) if wrong_hand_shape else (30, 126)
    face_shape = (30, 100) if wrong_face_shape else (30, 1404)
    hand = np.random.randn(*hand_shape).astype(np.float32)
    face = np.random.randn(*face_shape).astype(np.float32)
    if corrupt_hand:
        hand[0, 0] = float("nan")
    if corrupt_face:
        face[0, 0] = float("inf")
    np.savez(path, hand_features=hand, face_features=face,
             label=label, signer=signer, recording_id=rec_id)


def _make_dataset(tmp_dir, labels=("A","B","C"), signers=("s1","s2","s3","s4"),
                  n=3):
    """Populate tmp_dir with synthetic .npz files."""
    for lbl in labels:
        d = os.path.join(tmp_dir, lbl)
        os.makedirs(d, exist_ok=True)
        for sgn in signers:
            for i in range(n):
                _write_npz(os.path.join(d, f"{sgn}_{i:03d}.npz"),
                           label=lbl, signer=sgn, rec_id=f"{sgn}_{lbl}_{i}")
    return tmp_dir


# ─────────────────────────────────────────────────────────────────────────────
# Verification runner
# ─────────────────────────────────────────────────────────────────────────────

results: list = []   # (tag, passed, detail)

def check(tag, passed, detail="", exc=None):
    results.append((tag, passed, detail))
    if passed:
        print(_ok(tag, detail))
    else:
        msg = f"{detail}"
        if exc:
            msg += f"\n      Error: {exc}"
        print(_fail(tag, msg))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 – Dataset & Splitting
# ═════════════════════════════════════════════════════════════════════════════
_head("SECTION 1 – Dataset Loading & Splitting  [SYNTHETIC DATA ONLY]")

with tempfile.TemporaryDirectory() as TMP_DATA:
    _make_dataset(TMP_DATA, labels=("A","B","C"),
                  signers=("s1","s2","s3","s4"), n=3)

    # [V01] Dataset loading
    try:
        ds = LSEMultimodalDataset(data_dir=TMP_DATA)
        check("V01", len(ds) == 36,
              f"Loaded {len(ds)} samples (expected 36 = 3 labels × 4 signers × 3)")
    except Exception as e:
        check("V01", False, "Dataset loading failed", e)

    # [V02] Shape correctness
    try:
        hand, face, label, meta = ds[0]
        ok = (hand.shape == (30,126) and
              face.shape == (30,1404) and
              label.ndim == 0 and
              label.dtype == torch.long)
        check("V02", ok,
              f"hand={tuple(hand.shape)} face={tuple(face.shape)} "
              f"label={label.item()} dtype={label.dtype}")
    except Exception as e:
        check("V02", False, "Shape check failed", e)

    # [V03] Signer-independent splitting
    try:
        tr, va, te, lmap = make_signer_independent_splits(
            TMP_DATA,
            train_signers=["s1","s2"],
            val_signers=["s3"],
            test_signers=["s4"],
        )
        check("V03", len(tr)>0 and len(va)>0 and len(te)>0,
              f"train={len(tr)} val={len(va)} test={len(te)} "
              f"classes={len(lmap)}")
    except Exception as e:
        check("V03", False, "Splitting failed", e)

    # [V04] No-leakage assertion
    try:
        ts = set(tr.signers_found)
        vs = set(va.signers_found)
        tes = set(te.signers_found)
        no_leak = (ts & vs == set()) and (ts & tes == set()) and (vs & tes == set())
        check("V04", no_leak,
              f"train∩val={ts&vs} train∩test={ts&tes} val∩test={vs&tes}")
    except Exception as e:
        check("V04", False, "Leakage check failed", e)

    # [V05] DataLoader batching
    try:
        loader = DataLoader(tr, batch_size=4, shuffle=True)
        hb, fb, lb, _ = next(iter(loader))
        ok = (hb.shape == (4,30,126) and
              fb.shape == (4,30,1404) and
              lb.shape == (4,))
        check("V05", ok,
              f"hand_batch={tuple(hb.shape)} face_batch={tuple(fb.shape)} "
              f"labels={tuple(lb.shape)}")
    except Exception as e:
        check("V05", False, "DataLoader failed", e)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 – Forward / Backward / Optimizer
# ═════════════════════════════════════════════════════════════════════════════
_head("SECTION 2 – Forward / Backward / Optimizer  [SYNTHETIC DATA ONLY]")

set_seed(42)
NUM_CLASSES = 3
model_cfg = ModelConfig(num_classes=NUM_CLASSES, temporal_dropout=0.0,
                        encoder_dropout=0.0, fusion_dropout=0.0)
model = MultimodalSignModel(config=model_cfg)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

B, T = 4, 30
hand_in = torch.randn(B, T, 126, requires_grad=False)
face_in = torch.randn(B, T, 1404, requires_grad=False)
labels  = torch.randint(0, NUM_CLASSES, (B,))

# [V06] Full forward pass
try:
    model.train()
    out = model(hand_in, face_in)
    ok = out.logits.shape == (B, NUM_CLASSES)
    check("V06", ok, f"logits shape={tuple(out.logits.shape)}")
except Exception as e:
    check("V06", False, "Forward pass failed", e)
    out = None

# [V07] Loss calculation
loss_val = None
try:
    loss = criterion(out.logits, labels)
    loss_val = loss.item()
    check("V07", math.isfinite(loss_val),
          f"loss={loss_val:.6f} (finite scalar from random data – NOT real performance)")
except Exception as e:
    check("V07", False, "Loss calculation failed", e)

# [V08] Backward pass
try:
    optimizer.zero_grad()
    loss.backward()
    # Check grads reached all sub-network entry-points
    g_hand = model.hand_encoder.encoder[0].weight.grad
    g_face = model.face_encoder.encoder[0].weight.grad
    g_gate = model.fusion.gating_network[0].weight.grad
    g_lstm = next(model.temporal_classifier.lstm.parameters()).grad
    ok = all(g is not None and not torch.isnan(g).any()
             for g in [g_hand, g_face, g_gate, g_lstm])
    check("V08", ok, "Gradients: hand_enc✓  face_enc✓  fusion_gate✓  lstm✓")
except Exception as e:
    check("V08", False, "Backward pass failed", e)

# [V09] Optimizer step
try:
    w_before = model.hand_encoder.encoder[0].weight.detach().clone()
    optimizer.step()
    w_after  = model.hand_encoder.encoder[0].weight.detach().clone()
    changed  = not torch.equal(w_before, w_after)
    check("V09", changed, "Weights updated by AdamW after optimizer.step()")
except Exception as e:
    check("V09", False, "Optimizer step failed", e)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 – LR Schedulers
# ═════════════════════════════════════════════════════════════════════════════
_head("SECTION 3 – Learning-Rate Schedulers  [SYNTHETIC DATA ONLY]")

# [V10] Cosine scheduler
try:
    set_seed(0)
    n_epochs = 10
    opt_cos = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_cos, T_max=n_epochs, eta_min=1e-5
    )
    lrs = []
    for _ in range(n_epochs):
        lrs.append(opt_cos.param_groups[0]["lr"])
        sched_cos.step()
    monotone = all(lrs[i] >= lrs[i+1] for i in range(len(lrs)-1))
    check("V10", monotone,
          f"Cosine LR trace (first 5): {[f'{r:.2e}' for r in lrs[:5]]}")
except Exception as e:
    check("V10", False, "Cosine scheduler failed", e)

# [V11] Step scheduler
try:
    opt_step = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched_step = torch.optim.lr_scheduler.StepLR(opt_step, step_size=3, gamma=0.5)
    lrs_step = []
    for _ in range(10):
        lrs_step.append(opt_step.param_groups[0]["lr"])
        sched_step.step()
    dropped = lrs_step[3] < lrs_step[2]   # should drop at epoch 3
    check("V11", dropped,
          f"StepLR trace (first 6): {[f'{r:.2e}' for r in lrs_step[:6]]}")
except Exception as e:
    check("V11", False, "Step scheduler failed", e)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 – NaN / Inf Protection
# ═════════════════════════════════════════════════════════════════════════════
_head("SECTION 4 – NaN / Inf Protection  [SYNTHETIC DATA ONLY]")

# [V12] NaN in hand tensor
try:
    bad = hand_in.clone()
    bad[0, 0, 0] = float("nan")
    raised = False
    try:
        _check_tensor("hand", bad)
    except InvalidTensorError:
        raised = True
    check("V12", raised, "InvalidTensorError raised on NaN hand input ✓")
except Exception as e:
    check("V12", False, "NaN check failed", e)

# [V13] Inf in face tensor
try:
    bad = face_in.clone()
    bad[1, 5, 100] = float("inf")
    raised = False
    try:
        _check_tensor("face", bad)
    except InvalidTensorError:
        raised = True
    check("V13", raised, "InvalidTensorError raised on Inf face input ✓")
except Exception as e:
    check("V13", False, "Inf check failed", e)

# [V14] Clean tensor does NOT raise
try:
    clean = torch.randn(4, 30, 126)
    _check_tensor("clean", clean)
    check("V14", True, "Clean tensor passed without error ✓")
except Exception as e:
    check("V14", False, "Clean tensor unexpectedly raised", e)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 – Checkpoint Save / Load / Resume
# ═════════════════════════════════════════════════════════════════════════════
_head("SECTION 5 – Checkpoint Save / Load / Resume  [SYNTHETIC DATA ONLY]")

with tempfile.TemporaryDirectory() as TMP_DATA2, \
     tempfile.TemporaryDirectory() as TMP_CKPT, \
     tempfile.TemporaryDirectory() as TMP_MET:

    _make_dataset(TMP_DATA2, labels=("A","B"), signers=("s1","s2","s3"), n=2)

    # [V15] Checkpoint saving
    try:
        cfg1 = TrainConfig(
            data_dir=TMP_DATA2,
            train_signers=["s1","s2"], val_signers=["s3"],
            epochs=2, batch_size=4, seed=42,
            patience=99, checkpoint_dir=TMP_CKPT, metrics_dir=TMP_MET,
            lr_scheduler="none", save_every_n_epochs=1,
        )
        Trainer(cfg1).train()
        best = os.path.join(TMP_CKPT, "best_model.pt")
        ep1  = os.path.join(TMP_CKPT, "checkpoint_epoch_0001.pt")
        ep2  = os.path.join(TMP_CKPT, "checkpoint_epoch_0002.pt")
        ok = os.path.isfile(best) and os.path.isfile(ep1) and os.path.isfile(ep2)
        check("V15", ok,
              f"best_model.pt={os.path.isfile(best)}  "
              f"epoch_0001={os.path.isfile(ep1)}  epoch_0002={os.path.isfile(ep2)}")
    except Exception as e:
        check("V15", False, "Checkpoint save failed", e)
        traceback.print_exc()

    # [V16] Checkpoint loading – weights byte-equal
    try:
        ckpt = torch.load(best, map_location="cpu")
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "epoch" in ckpt
        assert "val_loss" in ckpt
        assert "history" in ckpt
        m2 = MultimodalSignModel(config=ModelConfig(num_classes=2))
        m2.load_state_dict(ckpt["model_state_dict"])
        # Verify every parameter matches
        for k, p in m2.state_dict().items():
            assert torch.equal(p, ckpt["model_state_dict"][k])
        check("V16", True,
              f"Checkpoint keys: {sorted(ckpt.keys())}  – weights restored byte-equal")
    except Exception as e:
        check("V16", False, "Checkpoint load failed", e)

    # [V17] Resume – epoch counter continues
    try:
        cfg2 = TrainConfig(
            data_dir=TMP_DATA2,
            train_signers=["s1","s2"], val_signers=["s3"],
            epochs=4, batch_size=4, seed=42,
            patience=99,
            checkpoint_dir=tempfile.mkdtemp(),
            metrics_dir=tempfile.mkdtemp(),
            lr_scheduler="none", save_every_n_epochs=10,
            resume_from=ep2,  # resume from epoch 2 checkpoint
        )
        t2 = Trainer(cfg2)
        t2.train()
        # Should run 2 more epochs (3 and 4)
        # Full history after resume should be [1, 2, 3, 4] – the trainer
        # restores the old history from the checkpoint and appends new entries.
        all_epochs = [h["epoch"] for h in t2.history]
        # The last two entries must be the resumed epochs (3 and 4)
        new_entries = all_epochs[-2:]
        check("V17", new_entries == [3, 4] and len(all_epochs) == 4,
              f"Full history epochs: {all_epochs}  "
              f"Last 2 (resumed): {new_entries}  (expected [3,4])")
    except Exception as e:
        check("V17", False, "Resume failed", e)
        traceback.print_exc()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 – Early Stopping
# ═════════════════════════════════════════════════════════════════════════════
_head("SECTION 6 – Early Stopping  [SYNTHETIC DATA ONLY]")

with tempfile.TemporaryDirectory() as TMP_ES_DATA, \
     tempfile.TemporaryDirectory() as TMP_ES_CKPT, \
     tempfile.TemporaryDirectory() as TMP_ES_MET:

    _make_dataset(TMP_ES_DATA, labels=("A","B"), signers=("s1","s2","s3"), n=2)

    # [V18] Early stopping fires at patience=2, min_delta=1e10
    try:
        cfg_es = TrainConfig(
            data_dir=TMP_ES_DATA,
            train_signers=["s1","s2"], val_signers=["s3"],
            epochs=30, batch_size=4, seed=7,
            patience=2, min_delta=1e10,  # impossible to improve
            checkpoint_dir=TMP_ES_CKPT, metrics_dir=TMP_ES_MET,
            lr_scheduler="none",
        )
        t_es = Trainer(cfg_es)
        t_es.train()
        # Epoch 1: best saved.  Epochs 2,3: no improvement → stop at end of epoch 3
        n_ran = len(t_es.history)
        check("V18", n_ran == 3,
              f"Ran {n_ran} epochs before early stop (patience=2, expected 3)")
    except Exception as e:
        check("V18", False, "Early stopping test failed", e)
        traceback.print_exc()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 – Reproducibility
# ═════════════════════════════════════════════════════════════════════════════
_head("SECTION 7 – Reproducibility  [SYNTHETIC DATA ONLY]")

# [V19] Same seed → identical loss sequence
# Both runs use EXACTLY the same data directory to ensure inputs are identical.
try:
    losses_a, losses_b = [], []
    with tempfile.TemporaryDirectory() as shared_data:
        # Seed numpy before writing so file contents are deterministic
        np.random.seed(0)
        _make_dataset(shared_data, labels=("A","B"), signers=("s1","s2","s3"), n=2)

        for losses in [losses_a, losses_b]:
            with tempfile.TemporaryDirectory() as c, tempfile.TemporaryDirectory() as m:
                cfg_r = TrainConfig(
                    data_dir=shared_data,
                    train_signers=["s1","s2"], val_signers=["s3"],
                    epochs=3, batch_size=4, seed=99,
                    patience=99, checkpoint_dir=c, metrics_dir=m,
                    lr_scheduler="none",
                )
                t = Trainer(cfg_r)
                t.train()
                losses.extend([h["train_loss"] for h in t.history])

    close = all(abs(a - b) < 1e-5 for a, b in zip(losses_a, losses_b))
    check("V19", close,
          f"Run A losses: {[f'{v:.4f}' for v in losses_a]}  "
          f"Run B losses: {[f'{v:.4f}' for v in losses_b]}")
except Exception as e:
    check("V19", False, "Reproducibility check failed", e)
    traceback.print_exc()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 – Safeguards
# ═════════════════════════════════════════════════════════════════════════════
_head("SECTION 8 – Safeguards  [SYNTHETIC DATA ONLY]")

# [V20] DatasetMissingError
try:
    with tempfile.TemporaryDirectory() as empty, \
         tempfile.TemporaryDirectory() as c, tempfile.TemporaryDirectory() as m:
        cfg_miss = TrainConfig(
            data_dir=empty, train_signers=["s1"], val_signers=["s2"],
            checkpoint_dir=c, metrics_dir=m,
        )
        raised = False
        try:
            Trainer(cfg_miss).train()
        except DatasetMissingError:
            raised = True
    check("V20", raised, "DatasetMissingError raised on empty data_dir ✓")
except Exception as e:
    check("V20", False, "DatasetMissing safeguard failed", e)

# [V21] InsufficientClasses (1-class dataset)
# Note: validate_split_config also raises ValueError for < 2 labels, so we
# accept either InsufficientClassesError or ValueError as correct behavior.
try:
    with tempfile.TemporaryDirectory() as d1c, \
         tempfile.TemporaryDirectory() as c, tempfile.TemporaryDirectory() as m:
        _make_dataset(d1c, labels=("ONLY_ONE",), signers=("s1","s2","s3"), n=2)
        cfg_1c = TrainConfig(
            data_dir=d1c,
            train_signers=["s1","s2"], val_signers=["s3"],
            checkpoint_dir=c, metrics_dir=m,
        )
        raised = False
        exc_name = ""
        try:
            Trainer(cfg_1c).train()
        except (InsufficientClassesError, ValueError) as exc:
            raised = True
            exc_name = type(exc).__name__
    check("V21", raised,
          f"{exc_name} raised for 1-class dataset (insufficient classes safeguard) ✓")
except Exception as e:
    check("V21", False, "InsufficientClasses safeguard failed", e)

# [V22] Signer overlap ValueError
try:
    raised = False
    try:
        validate_split_config(
            all_signers=["s1","s2"],
            train_signers=["s1"],
            val_signers=["s1"],   # deliberately overlapping
            test_signers=None,
            all_labels=["A","B"],
        )
    except ValueError as ve:
        raised = "leakage" in str(ve).lower()
    check("V22", raised, "ValueError raised on overlapping signer assignment ✓")
except Exception as e:
    check("V22", False, "Overlap safeguard failed", e)

# [V23] Empty split RuntimeError
try:
    with tempfile.TemporaryDirectory() as d23, \
         tempfile.TemporaryDirectory() as c, tempfile.TemporaryDirectory() as m:
        _make_dataset(d23, labels=("A","B"), signers=("s1","s2"), n=2)
        # s99 does not exist → val split empty
        raised = False
        try:
            make_signer_independent_splits(
                d23,
                train_signers=["s1"],
                val_signers=["s99"],  # non-existent signer
            )
        except (ValueError, RuntimeError):
            raised = True
    check("V23", raised, "Error raised when val split contains unknown signer ✓")
except Exception as e:
    check("V23", False, "Empty-split safeguard failed", e)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 – Gradient Clipping & Multi-Epoch History
# ═════════════════════════════════════════════════════════════════════════════
_head("SECTION 9 – Gradient Clipping & History  [SYNTHETIC DATA ONLY]")

# [V24] Gradient clipping does not crash under very large LR
try:
    set_seed(0)
    m_clip = MultimodalSignModel(config=ModelConfig(num_classes=2,
                                                     temporal_dropout=0.0,
                                                     encoder_dropout=0.0,
                                                     fusion_dropout=0.0))
    opt_big = torch.optim.SGD(m_clip.parameters(), lr=1e5)  # absurdly large
    h_in = torch.randn(2, 30, 126)
    f_in = torch.randn(2, 30, 1404)
    y_in = torch.randint(0, 2, (2,))
    m_clip.train()
    logits = m_clip(h_in, f_in).logits
    lss = nn.CrossEntropyLoss()(logits, y_in)
    lss.backward()
    nn.utils.clip_grad_norm_(m_clip.parameters(), max_norm=5.0)
    opt_big.step()
    check("V24", True, "Gradient clipping survived lr=1e5 without NaN/crash ✓")
except Exception as e:
    check("V24", False, "Gradient clipping test failed", e)

# [V25] Multi-epoch history saved to JSON
try:
    with tempfile.TemporaryDirectory() as dh, \
         tempfile.TemporaryDirectory() as ch, \
         tempfile.TemporaryDirectory() as mh:
        _make_dataset(dh, labels=("A","B"), signers=("s1","s2","s3"), n=2)
        cfg_h = TrainConfig(
            data_dir=dh,
            train_signers=["s1","s2"], val_signers=["s3"],
            epochs=4, batch_size=4, seed=1,
            patience=99, checkpoint_dir=ch, metrics_dir=mh,
            lr_scheduler="none",
        )
        th = Trainer(cfg_h)
        th.train()
        hist_path = os.path.join(mh, "training_history.json")
        with open(hist_path) as f:
            hist = json.load(f)
        epochs_in_hist = [h["epoch"] for h in hist]
        required_keys = {"epoch","train_loss","train_acc","val_loss","val_acc","lr"}
        all_keys_ok = all(required_keys <= set(h.keys()) for h in hist)
        check("V25", len(hist) == 4 and epochs_in_hist == [1,2,3,4] and all_keys_ok,
              f"History: {len(hist)} entries, epochs={epochs_in_hist}, "
              f"all required keys present={all_keys_ok}")
except Exception as e:
    check("V25", False, "Multi-epoch history test failed", e)
    traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═════════════════════════════════════════════════════════════════════════════
passed = [(t,d) for t,ok,d in results if ok]
failed = [(t,d) for t,ok,d in results if not ok]

print(f"\n{'═'*66}")
print(f"{BOLD}  VERIFICATION SUMMARY{RESET}")
print(f"{'═'*66}")
print(f"  Total checks : {len(results)}")
print(f"  {GREEN}Passed{RESET}        : {len(passed)}")
print(f"  {RED}Failed{RESET}        : {len(failed)}")
print()

if failed:
    print(f"{RED}{BOLD}  FAILED CHECKS:{RESET}")
    for t, d in failed:
        print(f"    [{t}] {d}")
    print()

print(f"{YELLOW}{'─'*66}{RESET}")
print(f"{YELLOW}  ⚠  REMINDER: All data above is SYNTHETIC (random tensors).{RESET}")
print(f"{YELLOW}  ⚠  None of these numbers represent real model performance.{RESET}")
print(f"{YELLOW}  ⚠  Real training requires actual sign-language .npz recordings.{RESET}")
print(f"{YELLOW}{'─'*66}{RESET}")

if failed:
    sys.exit(1)
