"""Model shape tests: forward passes and generation produce sane tensors."""

import importlib.util
import os

import numpy as np
import torch

from flowforge.data import N_FEATURES, WINDOW
from flowforge.models import (BookDDPM, CondVAE, EMA, count_params,
                              sample_with_ema, set_seed, train_ddpm,
                              train_vae, vae_loss)


def _batch(n=8):
    set_seed(0)
    x = torch.randn(n, WINDOW, N_FEATURES)
    r = torch.randn(n, 2)
    return x, r


def test_vae_shapes():
    x, r = _batch()
    m = CondVAE()
    recon, mu, logvar = m(x, r)
    assert recon.shape == x.shape
    assert mu.shape == (x.shape[0], 16)
    assert logvar.shape == (x.shape[0], 16)
    loss, mse, kld = vae_loss(x, recon, mu, logvar)
    assert torch.isfinite(loss) and torch.isfinite(mse) and kld >= 0


def test_vae_generate_shape():
    _, r = _batch(n=4)
    m = CondVAE()
    g = m.generate(r, n_per_regime=3)
    assert g.shape == (12, WINDOW, N_FEATURES)
    assert torch.isfinite(g).all()


def test_ddpm_shapes():
    x, r = _batch()
    m = BookDDPM(t_steps=10)
    xc = x.permute(0, 2, 1)
    t = torch.randint(0, 10, (x.shape[0],))
    pred = m.net(xc, t, r)
    assert pred.shape == xc.shape
    loss = m.p_loss(xc, r)
    assert torch.isfinite(loss) and loss.item() > 0


def test_ddpm_sample_shape():
    _, r = _batch(n=4)
    m = BookDDPM(t_steps=10)
    g = m.sample(r, seed=0)
    assert g.shape == (4, WINDOW, N_FEATURES)
    assert torch.isfinite(g).all()


def test_models_are_small():
    assert count_params(CondVAE()) < 300_000
    assert count_params(BookDDPM()) < 500_000


def test_short_train_runs():
    x, r = _batch(n=32)
    vae = CondVAE()
    h = train_vae(vae, x, r, epochs=2, batch=16, seed=0, verbose=False)
    assert len(h) == 2 and all(np.isfinite(h))
    ddpm = BookDDPM(t_steps=10)
    h, ema = train_ddpm(ddpm, x, r, epochs=2, batch=16, seed=0, verbose=False)
    assert len(h) == 2 and all(np.isfinite(h)) and ema is None
    h, ema = train_ddpm(ddpm, x, r, epochs=2, batch=16, seed=0, verbose=False,
                        ema_decay=0.9)
    assert len(h) == 2 and ema is not None
    # EMA shadow weights track the model weights
    for n, p in ddpm.named_parameters():
        if p.requires_grad:
            assert torch.allclose(ema.shadow[n], p.detach(), atol=1e-3)


def test_ema_apply_restore_roundtrip():
    _, r = _batch(n=4)
    m = BookDDPM(t_steps=10)
    ema = EMA(m, decay=0.0)  # decay 0 -> shadow snaps to latest weights
    before = {n: p.detach().clone() for n, p in m.named_parameters()
              if p.requires_grad}
    ema.apply_to(m)
    ema.restore(m)
    for n, p in m.named_parameters():
        if p.requires_grad:
            assert torch.equal(p, before[n])


def test_sample_with_ema_shape():
    _, r = _batch(n=4)
    m = BookDDPM(t_steps=10)
    ema = EMA(m)
    g = sample_with_ema(m, ema, r, seed=0)
    assert g.shape == (4, WINDOW, N_FEATURES)
    assert torch.isfinite(g).all()


# ------------------------------------------- P04 part 1: ablation contract
# The oracle in examples/architecture_ablate.py computes receptive fields
# from declarative layer specs with plain integer arithmetic and shares
# no code with the models, so the expected values below are known
# independently of the implementation under test.

def _load_ablate():
    path = os.path.join(os.path.dirname(__file__), "..", "examples",
                        "architecture_ablate.py")
    spec = importlib.util.spec_from_file_location("architecture_ablate",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _conv(kernel=3, stride=1, dilation=1):
    return {"kind": "conv", "kernel": kernel, "stride": stride,
            "dilation": dilation}


def test_stage3_flowforge_04_part_1_oracle_known_answers():
    ab = _load_ablate()
    rf = ab.oracle_receptive_field
    assert rf([]) == 1
    assert rf([_conv()]) == 3
    assert rf([_conv(), _conv()]) == 5
    assert rf([_conv(dilation=2)]) == 5
    assert rf([_conv(kernel=5, dilation=2)]) == 9
    # stride grows the jump: the second conv sees a 2-spaced grid
    assert rf([_conv(stride=2), _conv()]) == 7
    # upsampling restores the jump before the next conv
    assert rf([_conv(stride=2), {"kind": "upsample", "factor": 2},
               _conv()]) == 5
    # a global mixing layer makes the field unbounded
    assert rf([{"kind": "global_mix"}]) is None
    assert ab.oracle_half_width([_conv(), _conv()]) == 2
    assert ab.oracle_influence_span([_conv(), _conv()], 10, 60) == (8, 12)
    assert ab.oracle_influence_span([_conv(), _conv()], 0, 60) == (0, 2)


def test_stage3_flowforge_04_part_1_baseline_spec_matches_denoiser():
    ab = _load_ablate()
    net = BookDDPM(run_id="p04-part1-baseline").net
    convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv1d)]
    # the spec is the Denoiser's actual signal path: in_conv, three
    # ResBlocks of two convolutions, and the output convolution
    assert len(convs) == len(ab.BASELINE_CNN_SPEC) == 8
    for m in convs:
        assert m.kernel_size == (3,) and m.stride == (1,)
        assert m.dilation == (1,) and m.padding == (1,)
    assert ab.oracle_receptive_field(ab.BASELINE_CNN_SPEC) == 17
    assert ab.oracle_half_width(ab.BASELINE_CNN_SPEC) == 8
    # an input at offset 0 reaches only the first 9 output positions
    assert ab.oracle_influence_span(ab.BASELINE_CNN_SPEC, 0, WINDOW) == (0, 8)


def test_stage3_flowforge_04_part_1_fixture_influence_matches_oracle():
    ab = _load_ablate()
    record = ab.capture_baseline()
    targets = record["empirical"]["targets"]
    bound = record["empirical"]["influence_relative_bound"]
    for key, span in (("30", [22, 38]), ("59", [51, 59])):
        got = targets[key]
        assert got["span"] == span
        # influence is demonstrably present where the field reaches
        assert got["within_span_max_grad"] > 0.05
        # and only the weak GroupNorm statistics coupling remains beyond
        # it: no meaningful global context through the input path
        assert got["beyond_to_within_ratio"] < bound
    assert record["theoretical"]["receptive_field"] == 17
    assert record["theoretical"]["full_window_context"] is False


def test_stage3_flowforge_04_part_1_conditioning_reaches_distant_offsets():
    ab = _load_ablate()
    record = ab.capture_baseline()
    # the regime vector is broadcast to every position, so conditioning
    # sensitivity at the last offset is the same order as the in-span
    # input influence, unlike the early-input influence measured above
    sens = record["empirical"]["conditioning_sensitivity_at_last_offset"]
    assert sens > 0.05


def test_stage3_flowforge_04_part_1_no_shape_or_padding_defects():
    from flowforge.models import Denoiser
    net = Denoiser()
    regime = torch.randn(2, 2)
    for length in (32, 59, 60, 61, 120):
        x = torch.randn(2, N_FEATURES, length)
        t = torch.zeros(2, dtype=torch.long)
        out = net(x, t, regime)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


def test_stage3_flowforge_04_part_1_candidate_spec_claims():
    ab = _load_ablate()
    report = ab.spec_report()
    # the TCN dilation cycle (1, 2, 4, 8, 16, 32) claims full-window
    # context: half width 128 covers the 59-offset end-to-end distance
    assert report["dilated_tcn"]["receptive_field"] == 257
    assert report["dilated_tcn"]["full_window_context"] is True
    # the compact U-Net claims wider context than the baseline (37 vs
    # 17) but not full-window context; the claim is recorded as such
    assert report["compact_unet"]["receptive_field"] == 37
    assert report["compact_unet"]["full_window_context"] is False
    # attention claims global context by construction
    assert report["attention"]["receptive_field"] is None
    assert report["attention"]["full_window_context"] is True
    # the param budget rule is pinned for part 2
    assert ab.within_param_budget(1000, 1000)
    assert ab.within_param_budget(1050, 1000)
    assert not ab.within_param_budget(1051, 1000)


def test_stage3_flowforge_04_part_1_baseline_capture_reproducible():
    ab = _load_ablate()
    first = ab.capture_baseline()
    second = ab.capture_baseline()
    # latency is the only field allowed to move between captures
    first.pop("latency")
    second.pop("latency")
    assert first == second
    assert first["capture_sha256"] == second["capture_sha256"]
    fresh = BookDDPM(run_id="p04-part1-baseline")
    assert first["model"]["params"] == count_params(fresh.net)
    assert first["model"]["param_budget_max"] > first["model"]["params"]
    assert len(first["model"]["init_state_sha256"]) == 64
