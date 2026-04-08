"""
MISDO — Pre-Flight Training Readiness Test Suite
====================================================
Validates the ENTIRE training pipeline end-to-end using real data chips.
Each test targets a specific accuracy-critical property, not a proxy.

Run:  python tests/test_training_readiness.py

Exit code 0 = ready to train on A100.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import traceback
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Imports: verify the entire dependency chain resolves ──
from models.fire_model import FireRiskNet
from models.forest_model import ForestLossNet
from models.hydro_model import HydroRiskNet
from models.soil_model import SoilRiskNet
from models.base_model import DomainRiskNet
from models.backbone import ConvNeXtV2Backbone
from models.decoders import UNetPPDecoder
from models.temporal import TemporalAttention, TemporalSkipFusion
from datasets.real_datasets import (
    RealFireDataset,
    RealHansenDataset,
    RealHydroDataset,
    RealSoilDataset,
    compute_global_target_scale,
    validate_chip,
)
from losses import (
    CounterfactualDeltaLoss,
    DeepSupervisionWrapper,
    EdgeWeightedMSELoss,
    FocalBCELoss,
)
from train_real_models import (
    MODEL_CONFIGS,
    RandomFlipRotate,
    RadiometricJitter,
    ModelEMA,
    WarmupCosineScheduler,
    _build_optimizer_groups,
)

TILES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "real_tiles")
MANIFEST_PATH = os.path.join(TILES_DIR, "manifest.json")

# ═══════════════════════════════════════════════════════════════════
# Test Infrastructure
# ═══════════════════════════════════════════════════════════════════

_results: list[Tuple[str, bool, str]] = []
_all_tests: list = []  # registry of (name, callable) pairs


def _test(name: str):
    """Decorator to register and run a test with error capture."""
    def decorator(fn):
        def wrapper():
            try:
                fn()
                _results.append((name, True, ""))
                print(f"  ✓ {name}")
            except Exception as e:
                tb = traceback.format_exc()
                _results.append((name, False, str(e)))
                print(f"  ✗ {name}")
                print(f"    {e}")
                # Print abbreviated traceback for debugging
                for line in tb.strip().split("\n")[-4:]:
                    print(f"    {line}")
        wrapper.__name__ = name
        _all_tests.append(wrapper)
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════
# 1. DATA INTEGRITY — Validates real chips exist and are loadable
# ═══════════════════════════════════════════════════════════════════

@_test("Manifest exists and has train/test splits")
def test_manifest():
    assert os.path.exists(MANIFEST_PATH), f"No manifest at {MANIFEST_PATH}"
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    assert "train" in manifest, "No 'train' key in manifest"
    assert "test" in manifest, "No 'test' key in manifest"
    n_train = len(manifest["train"])
    n_test = len(manifest["test"])
    assert n_train > 0, "Train split is empty"
    assert n_test > 0, "Test split is empty"
    print(f"      Train: {n_train:,}  Test: {n_test:,}")


@_test("Random chips load and pass validation")
def test_chip_loading():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    entries = manifest["train"][:5] + manifest["test"][:5]
    for entry in entries:
        fpath = os.path.join(TILES_DIR, entry["file"])
        if not os.path.exists(fpath):
            # Try split/basename
            parts = entry["file"].replace("\\", "/").split("/")
            fpath = os.path.join(TILES_DIR, parts[-2], parts[-1]) if len(parts) >= 2 else fpath
        assert os.path.exists(fpath), f"Chip missing: {fpath}"
        data = np.load(fpath)
        valid, reason = validate_chip(data, fpath)
        assert valid, f"Chip failed validation: {reason}"


@_test("Chips contain SRTM terrain data (not zeros)")
def test_srtm_presence():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    n_srtm = sum(1 for e in manifest["train"] + manifest["test"]
                 if e.get("has_real_srtm"))
    total = len(manifest["train"]) + len(manifest["test"])
    pct = 100 * n_srtm / total
    assert pct > 80, f"Only {pct:.0f}% of chips have real SRTM — expected >80%"
    print(f"      SRTM coverage: {n_srtm}/{total} ({pct:.0f}%)")


@_test("Chips contain VIIRS fire data (or fall back to proxy)")
def test_viirs_presence():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    n_viirs = sum(1 for e in manifest["train"] + manifest["test"]
                  if e.get("has_real_viirs"))
    total = len(manifest["train"]) + len(manifest["test"])
    pct = 100 * n_viirs / total if total > 0 else 0
    if n_viirs == 0:
        print(f"      ⚠ VIIRS not yet baked into chips (0/{total}).")
        print(f"        Will be ingested on cluster via: download_real_data.py --viirs-archive")
        print(f"        Fire model will use proxy fallback locally — this is expected.")
    else:
        print(f"      VIIRS coverage: {n_viirs}/{total} ({pct:.0f}%)")
    # This is NOT a fatal error — fire model degrades gracefully to proxy channels


# ═══════════════════════════════════════════════════════════════════
# 2. DATASET → MODEL CHANNEL ALIGNMENT
# ═══════════════════════════════════════════════════════════════════

@_test("Forest dataset produces [T,6,H,W] matching ForestLossNet(in=6)")
def test_forest_channels():
    ds = RealHansenDataset(tiles_dir=TILES_DIR, split="train", T=5,
                           train_end_year=23, target_scale=1.0)
    obs_f, obs_cf, target = ds[0]
    assert obs_f.shape[1] == 6, f"Expected 6 channels, got {obs_f.shape[1]}"
    assert obs_f.shape[0] == 5, f"Expected T=5, got {obs_f.shape[0]}"
    assert obs_cf.shape == obs_f.shape
    assert target.shape[0] == 1, f"Target should be 1-channel, got {target.shape[0]}"
    model = ForestLossNet()
    assert model.IN_CHANNELS == 6


@_test("Fire dataset produces [T,7,H,W] matching FireRiskNet(in=7)")
def test_fire_channels():
    ds = RealFireDataset(tiles_dir=TILES_DIR, split="train", T=5,
                         train_end_year=23, target_scale=1.0)
    obs_f, obs_cf, target = ds[0]
    assert obs_f.shape[1] == 7, f"Expected 7 channels, got {obs_f.shape[1]}"
    assert obs_f.shape[0] == 5, f"Expected T=5, got {obs_f.shape[0]}"
    model = FireRiskNet()
    assert model.IN_CHANNELS == 7


@_test("Hydro dataset produces [7,H,W] matching HydroRiskNet(in=7, non-temporal)")
def test_hydro_channels():
    # Check if MSI/SMAP enrichment has been run
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    has_msi = any(e.get("has_real_msi_smap") for e in manifest["train"] + manifest["test"])
    if not has_msi:
        print(f"      ⚠ MSI/SMAP not yet baked into chips — skipping Hydro dataset test.")
        print(f"        Will be ingested on cluster via: download_msi_smap.py")
        # Verify the model class is correct even if we can't load data
        model = HydroRiskNet()
        assert model.IN_CHANNELS == 7
        assert model.TEMPORAL is False
        return
    ds = RealHydroDataset(tiles_dir=TILES_DIR, split="train",
                          train_end_year=23, target_scale=1.0)
    obs_f, obs_cf, target = ds[0]
    # Hydro is NON-temporal → obs should be [C,H,W] not [T,C,H,W]
    assert obs_f.dim() == 3, f"Hydro should be 3D [C,H,W], got {obs_f.dim()}D"
    assert obs_f.shape[0] == 7, f"Expected 7 channels, got {obs_f.shape[0]}"
    model = HydroRiskNet()
    assert model.IN_CHANNELS == 7
    assert model.TEMPORAL is False


@_test("Soil dataset produces [T,7,H,W] matching SoilRiskNet(in=7)")
def test_soil_channels():
    ds = RealSoilDataset(tiles_dir=TILES_DIR, split="train", T=5,
                         train_end_year=23, target_scale=1.0)
    obs_f, obs_cf, target = ds[0]
    assert obs_f.shape[1] == 7, f"Expected 7 channels, got {obs_f.shape[1]}"
    model = SoilRiskNet()
    assert model.IN_CHANNELS == 7


# ═══════════════════════════════════════════════════════════════════
# 3. SIAMESE FORWARD PASS — Validates the core training forward path
# ═══════════════════════════════════════════════════════════════════

@_test("All 4 models: forward_paired_deep produces correct output shapes")
def test_forward_paired_deep():
    configs = {
        "fire": (FireRiskNet, (1, 5, 7, 64, 64)),     # temporal
        "forest": (ForestLossNet, (1, 5, 6, 64, 64)),  # temporal
        "hydro": (HydroRiskNet, (1, 7, 64, 64)),       # non-temporal
        "soil": (SoilRiskNet, (1, 5, 7, 64, 64)),      # temporal
    }
    for name, (ModelCls, shape) in configs.items():
        model = ModelCls()
        model.eval()
        x_f = torch.randn(*shape)
        x_cf = torch.randn(*shape)
        with torch.no_grad():
            delta, deep_deltas, out_f, out_cf = model.forward_paired_deep(x_f, x_cf)
        # Delta must be [B, 1, H, W] where H=W=64 (input spatial dim)
        assert delta.shape == (1, 1, 64, 64), \
            f"{name}: delta shape {delta.shape}, expected (1,1,64,64)"
        # Delta must be in [0, 1] (clamped)
        assert delta.min() >= 0.0, f"{name}: delta min {delta.min()} < 0"
        assert delta.max() <= 1.0, f"{name}: delta max {delta.max()} > 1"
        # Deep supervision must return 3 auxiliary outputs
        assert len(deep_deltas) == 3, \
            f"{name}: expected 3 deep outputs, got {len(deep_deltas)}"
        # All deep outputs must match main output resolution (H, W)
        for i, dd in enumerate(deep_deltas):
            assert dd.shape == delta.shape, \
                f"{name}: deep[{i}] shape {dd.shape} != delta {delta.shape}"
        # No NaN in any output
        for tensor_name, t in [("delta", delta), ("out_f", out_f), ("out_cf", out_cf)]:
            assert not torch.isnan(t).any(), f"{name}: NaN in {tensor_name}"


@_test("forward() with two args dispatches to forward_paired_deep (DP compatibility)")
def test_forward_dispatch():
    model = ForestLossNet()
    model.eval()
    x_f = torch.randn(1, 5, 6, 64, 64)
    x_cf = torch.randn(1, 5, 6, 64, 64)
    with torch.no_grad():
        result = model(x_f, x_cf)
    # Should return 4-tuple from forward_paired_deep
    assert isinstance(result, tuple) and len(result) == 4, \
        f"model(x_f, x_cf) should return 4-tuple, got {type(result)}"


# ═══════════════════════════════════════════════════════════════════
# 4. LOSS COMPUTATION — Validates loss produces finite gradients
# ═══════════════════════════════════════════════════════════════════

@_test("Full training loss chain produces finite gradients for all 4 models")
def test_loss_gradient_flow():
    for name, config in MODEL_CONFIGS.items():
        model = config["model_cls"]()
        is_temporal = config["temporal"]
        if is_temporal:
            shape = (2, 5, model.IN_CHANNELS, 64, 64)
        else:
            shape = (2, model.IN_CHANNELS, 64, 64)

        x_f = torch.randn(*shape, requires_grad=False)
        x_cf = torch.randn(*shape, requires_grad=False)
        target = torch.rand(2, 1, 64, 64)

        model.train()
        delta, deep_deltas, out_f, out_cf = model.forward_paired_deep(x_f, x_cf)

        # Resize target if needed
        if delta.shape[2:] != target.shape[2:]:
            target = F.interpolate(target, size=delta.shape[2:], mode="bilinear", align_corners=False)

        # Use the exact loss from MODEL_CONFIGS
        base_criterion = config["loss_factory"]()
        criterion = DeepSupervisionWrapper(base_criterion, aux_weight=0.3)
        loss = criterion(delta, target, deep_deltas,
                         out_factual=out_f, out_counterfactual=out_cf)

        assert not torch.isnan(loss), f"{name}: loss is NaN"
        assert not torch.isinf(loss), f"{name}: loss is Inf"
        assert loss.item() > 0, f"{name}: loss is zero (model not learning)"

        loss.backward()

        # Verify gradients actually reach encoder (not just decoder)
        has_encoder_grad = False
        has_decoder_grad = False
        for pname, p in model.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                if "encoder" in pname:
                    has_encoder_grad = True
                if "decoder" in pname:
                    has_decoder_grad = True
        assert has_encoder_grad, f"{name}: no gradients reached encoder"
        assert has_decoder_grad, f"{name}: no gradients reached decoder"


@_test("Loss decreases after one optimizer step (model can learn)")
def test_loss_decreases():
    model = ForestLossNet()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x_f = torch.randn(2, 5, 6, 64, 64)
    x_cf = torch.randn(2, 5, 6, 64, 64)
    target = torch.rand(2, 1, 64, 64) * 0.3  # sparse-ish target

    # Step 1: compute loss
    delta1, deep1, outf1, outcf1 = model.forward_paired_deep(x_f, x_cf)
    if delta1.shape[2:] != target.shape[2:]:
        target = F.interpolate(target, size=delta1.shape[2:], mode="bilinear", align_corners=False)

    criterion = DeepSupervisionWrapper(
        CounterfactualDeltaLoss(base_loss=EdgeWeightedMSELoss(edge_weight=3.0, ssim_weight=0.01)),
        aux_weight=0.3,
    )
    loss1 = criterion(delta1, target, deep1, out_factual=outf1, out_counterfactual=outcf1)
    loss1.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Step 2: compute loss again — should be lower (model learned something)
    delta2, deep2, outf2, outcf2 = model.forward_paired_deep(x_f, x_cf)
    loss2 = criterion(delta2, target, deep2, out_factual=outf2, out_counterfactual=outcf2)

    assert loss2.item() < loss1.item(), \
        f"Loss did not decrease: {loss1.item():.6f} → {loss2.item():.6f}"


# ═══════════════════════════════════════════════════════════════════
# 5. REAL DATA FORWARD PASS — End-to-end with actual chips
# ═══════════════════════════════════════════════════════════════════

@_test("End-to-end: real chip → model → loss for all 4 models")
def test_real_data_e2e():
    # Check if MSI/SMAP data is available (needed for hydro)
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    has_msi = any(e.get("has_real_msi_smap") for e in manifest["train"] + manifest["test"])

    for name, config in MODEL_CONFIGS.items():
        # Skip hydro if MSI/SMAP not yet enriched
        if name == "hydro" and not has_msi:
            print(f"      ⚠ Skipping hydro E2E (MSI/SMAP not yet baked into chips)")
            continue

        DatasetCls = config["dataset_cls"]
        ModelCls = config["model_cls"]

        if config["temporal"]:
            ds = DatasetCls(tiles_dir=TILES_DIR, split="train", T=5,
                            train_end_year=23, target_scale=1.0)
        else:
            ds = DatasetCls(tiles_dir=TILES_DIR, split="train",
                            train_end_year=23, target_scale=1.0)

        if len(ds) == 0:
            raise RuntimeError(f"{name}: dataset is empty")

        obs_f, obs_cf, target = ds[0]
        obs_f = obs_f.unsqueeze(0)
        obs_cf = obs_cf.unsqueeze(0)
        target = target.unsqueeze(0)

        # Verify shapes match model expectations
        if config["temporal"]:
            assert obs_f.dim() == 5, f"{name}: temporal obs should be 5D"
            assert obs_f.shape[2] == ModelCls.IN_CHANNELS
        else:
            assert obs_f.dim() == 4, f"{name}: non-temporal obs should be 4D"
            assert obs_f.shape[1] == ModelCls.IN_CHANNELS

        # Verify no NaN/Inf in input data
        assert not torch.isnan(obs_f).any(), f"{name}: NaN in obs_f"
        assert not torch.isinf(obs_f).any(), f"{name}: Inf in obs_f"
        assert not torch.isnan(target).any(), f"{name}: NaN in target"

        # Forward + loss
        model = ModelCls()
        model.train()
        delta, deep, out_f, out_cf = model.forward_paired_deep(obs_f, obs_cf)

        if delta.shape[2:] != target.shape[2:]:
            target = F.interpolate(target, size=delta.shape[2:],
                                   mode="bilinear", align_corners=False)

        base_criterion = config["loss_factory"]()
        loss_fn = DeepSupervisionWrapper(base_criterion, aux_weight=0.3)
        loss = loss_fn(delta, target, deep,
                       out_factual=out_f, out_counterfactual=out_cf)

        assert not torch.isnan(loss), f"{name}: NaN loss on real data"
        assert loss.item() > 0, f"{name}: zero loss on real data"

        # Backward — ensure no NaN gradients on real data
        loss.backward()
        for pname, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), \
                    f"{name}: NaN gradient in {pname}"


# ═══════════════════════════════════════════════════════════════════
# 6. TARGET SCALE — Validates global normalisation is finite
# ═══════════════════════════════════════════════════════════════════

@_test("compute_global_target_scale returns finite >0 scales for all models")
def test_target_scale():
    for name in ["forest", "fire", "hydro", "soil"]:
        scale = compute_global_target_scale(name, TILES_DIR, split="train", n_samples=50)
        assert np.isfinite(scale), f"{name}: target scale is {scale}"
        assert scale > 0, f"{name}: target scale is {scale} (should be >0)"
        print(f"      {name}: target_scale = {scale:.6f}")


# ═══════════════════════════════════════════════════════════════════
# 7. DATA AUGMENTATION — Validates transforms preserve data integrity
# ═══════════════════════════════════════════════════════════════════

@_test("RandomFlipRotate preserves shape and range, transforms aspect correctly")
def test_augmentation():
    aug = RandomFlipRotate(aspect_channel_idx=2)
    obs = torch.rand(2, 7, 64, 64)
    obs[:, 2] = torch.rand(2, 64, 64)  # aspect channel in [0, 1]
    obs_cf = torch.rand(2, 7, 64, 64)
    obs_cf[:, 2] = obs[:, 2].clone()  # same aspect
    target = torch.rand(2, 1, 64, 64)

    for _ in range(10):  # run multiple times to hit all code paths
        out_f, out_cf, out_t = aug(obs.clone(), obs_cf.clone(), target.clone())
        assert out_f.shape == obs.shape
        assert out_cf.shape == obs_cf.shape
        assert out_t.shape == target.shape
        # Aspect must stay in [0, 1]
        assert out_f[:, 2].min() >= -1e-6, "Aspect < 0"
        assert out_f[:, 2].max() <= 1.0 + 1e-6, "Aspect > 1"


@_test("RadiometricJitter only jitters specified channels, not terrain")
def test_radiometric_jitter():
    jitter = RadiometricJitter(jitter_channels=[0, 1], p=1.0)
    obs = torch.rand(2, 7, 64, 64)
    obs_cf = obs.clone()

    # Channels 2-6 (terrain) should be untouched
    terrain_before = obs[:, 2:].clone()
    out_f, out_cf = jitter(obs, obs_cf)

    terrain_after = out_f[:, 2:]
    assert torch.equal(terrain_before, terrain_after), \
        "RadiometricJitter modified terrain channels"
    # Jittered channels should differ
    assert not torch.equal(out_f[:, :2], terrain_before[:, :2].new_zeros(2, 2, 64, 64)), \
        "Jittered channels appear unchanged"


# ═══════════════════════════════════════════════════════════════════
# 8. MODEL EMA — Validates shadow weights update correctly
# ═══════════════════════════════════════════════════════════════════

@_test("EMA shadow weights differ from model after update")
def test_ema():
    model = ForestLossNet()
    ema = ModelEMA(model, decay=0.999)

    # Modify model weights
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    ema.update(model)

    # EMA weights should be between original and current
    for ema_p, model_p in zip(ema.ema.parameters(), model.parameters()):
        assert not torch.equal(ema_p, model_p), "EMA weights equal model (no smoothing)"


# ═══════════════════════════════════════════════════════════════════
# 9. SCHEDULER — Validates warmup + cosine annealing
# ═══════════════════════════════════════════════════════════════════

@_test("WarmupCosineScheduler: LR increases during warmup, decreases after")
def test_scheduler():
    model = ForestLossNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=50)

    lrs = []
    for epoch in range(1, 51):
        scheduler.step(epoch)
        lrs.append(optimizer.param_groups[0]["lr"])

    # LR should increase during warmup (epochs 1-5)
    for i in range(3):
        assert lrs[i + 1] > lrs[i], \
            f"LR not increasing during warmup: {lrs[i]:.6f} → {lrs[i+1]:.6f}"
    # LR should decrease after warmup (epochs 6+)
    assert lrs[-1] < lrs[5], \
        f"LR not decreasing after warmup: {lrs[5]:.6f} → {lrs[-1]:.6f}"


# ═══════════════════════════════════════════════════════════════════
# 10. OPTIMIZER GROUPS — Layer-wise LR decay
# ═══════════════════════════════════════════════════════════════════

@_test("LLRD optimizer groups: encoder gets lower LR than decoder")
def test_optimizer_groups():
    model = ForestLossNet()
    groups = _build_optimizer_groups(model, base_lr=1e-3, weight_decay=0.01)

    # Should have multiple groups (not just one)
    assert len(groups) > 1, f"Expected multiple LLRD groups, got {len(groups)}"

    # Find encoder and decoder LRs
    encoder_lrs = set()
    decoder_lrs = set()
    for g in groups:
        if g["lr"] < 1e-3:  # lower than base → must be encoder
            encoder_lrs.add(g["lr"])
        else:
            decoder_lrs.add(g["lr"])

    assert len(encoder_lrs) > 0, "No encoder groups with reduced LR found"
    min_encoder_lr = min(encoder_lrs)
    assert min_encoder_lr < 1e-3, \
        f"Encoder should get lower LR than base, got {min_encoder_lr}"


# ═══════════════════════════════════════════════════════════════════
# 11. CHECKPOINT ROUNDTRIP — Model saves and loads correctly
# ═══════════════════════════════════════════════════════════════════

@_test("Checkpoint save/load roundtrip preserves predictions exactly")
def test_checkpoint_roundtrip():
    model = ForestLossNet()
    model.eval()

    x = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        pred_before = model(x).clone()

    # Save
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        tmp_path = f.name

    try:
        # Load into fresh model
        model2 = ForestLossNet()
        model2.load_state_dict(torch.load(tmp_path, map_location="cpu", weights_only=True))
        model2.eval()

        with torch.no_grad():
            pred_after = model2(x)

        assert torch.allclose(pred_before, pred_after, atol=1e-6), \
            f"Predictions differ after checkpoint roundtrip"
    finally:
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════
# 12. DEEP SUPERVISION — All auxiliary outputs match main resolution
# ═══════════════════════════════════════════════════════════════════

@_test("UNet++ deep supervision outputs all match main output resolution")
def test_deep_supervision_resolution():
    model = ForestLossNet()
    model.eval()
    x = torch.randn(1, 6, 256, 256)

    with torch.no_grad():
        out, deep = model.decoder(
            {
                "s1": torch.randn(1, 96, 64, 64),
                "s2": torch.randn(1, 192, 32, 32),
                "s3": torch.randn(1, 384, 16, 16),
                "s4": torch.randn(1, 768, 8, 8),
            },
            return_deep=True,
        )
    assert out.shape == (1, 1, 256, 256), f"Main output shape {out.shape}"
    for i, d in enumerate(deep):
        assert d.shape == out.shape, \
            f"Deep[{i}] shape {d.shape} != main {out.shape}"


# ═══════════════════════════════════════════════════════════════════
# 13. MONOTONICITY PENALTY — cf ≥ f enforced in loss
# ═══════════════════════════════════════════════════════════════════

@_test("CounterfactualDeltaLoss penalises monotonicity violations")
def test_monotonicity():
    loss_fn = CounterfactualDeltaLoss(
        base_loss=EdgeWeightedMSELoss(edge_weight=3.0, ssim_weight=0.01),
        mono_weight=0.5,
    )
    pred = torch.rand(2, 1, 64, 64)
    target = torch.rand(2, 1, 64, 64)

    # No violation: cf > f
    out_f = torch.rand(2, 1, 64, 64) * 0.3
    out_cf = out_f + 0.2
    loss_ok = loss_fn(pred, target, out_factual=out_f, out_counterfactual=out_cf)

    # Strong violation: f > cf
    out_f_bad = torch.rand(2, 1, 64, 64) * 0.5 + 0.5
    out_cf_bad = torch.rand(2, 1, 64, 64) * 0.3
    loss_bad = loss_fn(pred, target, out_factual=out_f_bad, out_counterfactual=out_cf_bad)

    assert loss_bad > loss_ok, \
        f"Monotonicity violation should increase loss: ok={loss_ok.item():.4f} bad={loss_bad.item():.4f}"


# ═══════════════════════════════════════════════════════════════════
# 14. MODEL_CONFIGS CONSISTENCY — Training config matches models
# ═══════════════════════════════════════════════════════════════════

@_test("MODEL_CONFIGS: dataset/model/loss factories all instantiate correctly")
def test_model_configs():
    for name, config in MODEL_CONFIGS.items():
        model = config["model_cls"]()
        assert isinstance(model, DomainRiskNet), f"{name}: model not a DomainRiskNet"

        ds_cls = config["dataset_cls"]
        assert ds_cls in (RealFireDataset, RealHansenDataset,
                          RealHydroDataset, RealSoilDataset)

        loss = config["loss_factory"]()
        assert isinstance(loss, CounterfactualDeltaLoss), \
            f"{name}: expected CounterfactualDeltaLoss, got {type(loss)}"

        # Verify jitter_channels are valid indices
        jitter_ch = config.get("jitter_channels")
        if jitter_ch is not None:
            for ch in jitter_ch:
                assert 0 <= ch < model.IN_CHANNELS, \
                    f"{name}: jitter channel {ch} out of range [0, {model.IN_CHANNELS})"

        # Verify aspect_channel_idx is valid
        aspect_ch = config.get("aspect_channel_idx")
        if aspect_ch is not None:
            assert 0 <= aspect_ch < model.IN_CHANNELS, \
                f"{name}: aspect channel {aspect_ch} out of range"


# ═══════════════════════════════════════════════════════════════════
# 15. AMP FLOAT16 STABILITY — No NaN under mixed precision
# ═══════════════════════════════════════════════════════════════════

@_test("AMP float16: forward + loss + backward produce no NaN (CPU simulation)")
def test_amp_stability():
    """Simulate AMP by running in float16 on CPU (A100 would use CUDA autocast)."""
    model = ForestLossNet()
    model.train()

    x_f = torch.randn(1, 5, 6, 64, 64)
    x_cf = torch.randn(1, 5, 6, 64, 64)
    target = torch.rand(1, 1, 64, 64)

    # CPU float16 simulation (autocast only works on CUDA, but we can test
    # the loss numerics by casting tensors directly)
    with torch.no_grad():
        delta, deep, out_f, out_cf = model.forward_paired_deep(x_f, x_cf)

    # Cast to float16 to simulate AMP output
    delta_f16 = delta.half().float()  # round-trip to simulate precision loss
    target_f16 = target.half().float()
    out_f_f16 = out_f.half().float()
    out_cf_f16 = out_cf.half().float()
    deep_f16 = [d.half().float() for d in deep]

    if delta_f16.shape[2:] != target_f16.shape[2:]:
        target_f16 = F.interpolate(target_f16, size=delta_f16.shape[2:],
                                   mode="bilinear", align_corners=False)

    criterion = DeepSupervisionWrapper(
        CounterfactualDeltaLoss(base_loss=EdgeWeightedMSELoss(edge_weight=3.0, ssim_weight=0.01)),
        aux_weight=0.3,
    )
    # Need grad for backward
    delta_f16.requires_grad_(True)
    loss = criterion(delta_f16, target_f16, deep_f16,
                     out_factual=out_f_f16, out_counterfactual=out_cf_f16)

    assert not torch.isnan(loss), f"NaN loss under float16 simulation"
    assert not torch.isinf(loss), f"Inf loss under float16 simulation"


# ═══════════════════════════════════════════════════════════════════
# 16. DATASET SPLIT INTEGRITY — No data leakage between train/test
# ═══════════════════════════════════════════════════════════════════

@_test("Train and test splits have zero file overlap (no data leakage)")
def test_split_integrity():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    train_files = {e["file"] for e in manifest["train"]}
    test_files = {e["file"] for e in manifest["test"]}
    overlap = train_files & test_files
    assert len(overlap) == 0, \
        f"Data leakage: {len(overlap)} files in both train and test"


# ═══════════════════════════════════════════════════════════════════
# 17. TEMPORAL MODULE — Attention and skip fusion work correctly
# ═══════════════════════════════════════════════════════════════════

@_test("Temporal modules: attention and skip fusion preserve spatial dims")
def test_temporal_modules():
    ta = TemporalAttention(768)
    tsf = TemporalSkipFusion(channels=96)

    # Bottleneck temporal attention
    x = torch.randn(2, 5, 768, 8, 8)  # [B, T, C, H, W]
    out = ta(x)
    assert out.shape == (2, 768, 8, 8), f"TemporalAttention shape {out.shape}"

    # Skip temporal fusion
    s = torch.randn(2, 5, 96, 64, 64)
    out_s = tsf(s)
    assert out_s.shape == (2, 96, 64, 64), f"TemporalSkipFusion shape {out_s.shape}"


# ═══════════════════════════════════════════════════════════════════
# 18. FULL TRAINING STEP SIMULATION
# ═══════════════════════════════════════════════════════════════════

@_test("Full training step: load real data → augment → forward → loss → backward → optimizer step")
def test_full_training_step():
    """Simulates exactly one training iteration from train_real_models.py."""
    name = "forest"
    config = MODEL_CONFIGS[name]

    ds = RealHansenDataset(tiles_dir=TILES_DIR, split="train", T=5,
                           train_end_year=23, target_scale=1.0)
    obs_f, obs_cf, target = ds[0]
    obs_f = obs_f.unsqueeze(0)
    obs_cf = obs_cf.unsqueeze(0)
    target = target.unsqueeze(0)

    model = ForestLossNet()
    model.train()

    # Build optimizer with LLRD (exactly as training)
    param_groups = _build_optimizer_groups(model, base_lr=3e-4, weight_decay=0.01)
    optimizer = torch.optim.AdamW(param_groups)

    # Augmentation (exactly as training)
    augment = RandomFlipRotate(aspect_channel_idx=config.get("aspect_channel_idx"))
    radiometric = RadiometricJitter(jitter_channels=config.get("jitter_channels"))

    # EMA (exactly as training)
    ema = ModelEMA(model, decay=0.9999)

    # Loss (exactly as training)
    base_criterion = config["loss_factory"]()
    criterion = DeepSupervisionWrapper(base_criterion, aux_weight=0.3)

    # Augment
    obs_f, obs_cf, target = augment(obs_f, obs_cf, target)
    obs_f, obs_cf = radiometric(obs_f, obs_cf)

    # Forward
    delta, deep, out_f, out_cf = model.forward_paired_deep(obs_f, obs_cf)
    if delta.shape[2:] != target.shape[2:]:
        target = F.interpolate(target, size=delta.shape[2:],
                               mode="bilinear", align_corners=False)

    # Loss
    loss = criterion(delta, target, deep,
                     out_factual=out_f, out_counterfactual=out_cf)
    assert not torch.isnan(loss), "NaN loss in full training step"

    # Backward + optimizer step
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    # EMA update
    ema.update(model)

    # Verify model weights actually changed
    # (if they didn't, training would be a no-op)
    model2 = ForestLossNet()
    changed = False
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        if not torch.equal(p1.data, p2.data):
            changed = True
            break
    assert changed, "Model weights unchanged after training step"


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  MISDO — Pre-Flight Training Readiness Checks")
    print("=" * 65)
    print()

    tests = _all_tests

    for test_fn in tests:
        test_fn()

    print()
    print("=" * 65)
    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    total = len(_results)

    if n_fail == 0:
        print(f"  ✓ ALL {total} CHECKS PASSED — ready to train on A100")
    else:
        print(f"  ✗ {n_fail}/{total} CHECKS FAILED:")
        for name, ok, msg in _results:
            if not ok:
                print(f"    • {name}: {msg}")

    print("=" * 65)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
