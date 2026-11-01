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


# ------------------------------------------- P04 part 2: candidate archs
# The candidates are built to the declarative specs pinned in part 1 and
# measured under the part 1 fixture protocol (run ID p04-part1-baseline
# streams for both init and the fixture input), so every number below is
# paired with the baseline measurements in the part 1 tests.

def _p04_fixture():
    from flowforge.models import derive_seed, scoped_rng
    ab = _load_ablate()
    run_id = ab.BASELINE_RUN_ID
    with scoped_rng(derive_seed(run_id, "p04:fixture")):
        x = torch.randn(2, N_FEATURES, WINDOW)
        regime = torch.randn(2, 2)
    t = torch.full((2,), 3, dtype=torch.long)
    return ab, run_id, x, t, regime


def test_stage3_flowforge_04_part_2_dilated_tcn_spec_matches_implementation():
    ab = _load_ablate()
    net = BookDDPM(arch="dilated_tcn",
                   run_id=ab.BASELINE_RUN_ID).net
    convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv1d)]
    spec_dilations = [layer["dilation"] for layer in ab.DILATED_TCN_SPEC
                      if layer["kind"] == "conv"]
    # in/out convolutions plus six blocks of two convolutions, block
    # dilations following the explicit cycle 1, 2, 4, 8, 16, 32
    assert len(convs) == len(spec_dilations) == 14
    assert [m.dilation[0] for m in convs] == spec_dilations
    assert all(m.kernel_size == (3,) and m.stride == (1,) for m in convs)
    # padding tracks dilation, so every convolution preserves length
    assert all(m.padding == m.dilation for m in convs)


def test_stage3_flowforge_04_part_2_dilated_tcn_influence_beyond_baseline_span():
    ab, run_id, x, t, regime = _p04_fixture()
    tcn = BookDDPM(arch="dilated_tcn", run_id=run_id).net
    cnn = BookDDPM(run_id=run_id).net
    target = WINDOW - 1
    tcn_profile = ab.gradient_influence(tcn, x, t, regime, target)
    cnn_profile = ab.gradient_influence(cnn, x, t, regime, target)
    # offsets 19, 29, 41 are far outside the baseline span for this
    # target (span ends at offset 51 looking back 8) but inside the
    # TCN span: the TCN gradient there is several times the baseline
    # gradient at the same offsets, so early-window signal demonstrably
    # reaches the late prediction through the dilated feature path
    for off in (19, 29, 41):
        assert tcn_profile[off] > 0.012
        assert tcn_profile[off] > 2.0 * cnn_profile[off]
    # recorded honestly: at init the extreme corner (offset 0, distance
    # 59) still sits near the normalization-leakage level even though
    # the dilated path connects it; the demonstrated influence above is
    # at distances up to 40
    assert tcn_profile[0] < tcn_profile[29]


def test_stage3_flowforge_04_part_2_compact_unet_spec_matches_implementation():
    ab = _load_ablate()
    net = BookDDPM(arch="compact_unet",
                   run_id=ab.BASELINE_RUN_ID).net
    convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv1d)]
    spec_shapes = [(layer["kernel"], layer["stride"])
                   for layer in ab.COMPACT_UNET_SPEC
                   if layer["kind"] == "conv"]
    assert len(convs) == len(spec_shapes) == 9
    assert [(m.kernel_size[0], m.stride[0]) for m in convs] == spec_shapes
    # the claim is a span of half width 18, not full-window context
    assert ab.oracle_influence_span(ab.COMPACT_UNET_SPEC, 0, WINDOW) \
        == (0, 18)
    assert ab.oracle_influence_span(ab.COMPACT_UNET_SPEC, WINDOW - 1,
                                    WINDOW) == (WINDOW - 19, WINDOW - 1)


def test_stage3_flowforge_04_part_2_compact_unet_influence_matches_its_span():
    ab, run_id, x, t, regime = _p04_fixture()
    net = BookDDPM(arch="compact_unet", run_id=run_id).net
    bound = ab.INFLUENCE_RELATIVE_BOUND
    for target, span in ((WINDOW - 1, [41, 59]), (WINDOW // 2, [12, 48])):
        profile = ab.gradient_influence(net, x, t, regime, target)
        got = ab.influence_summary(profile, target, 18)
        assert got["span"] == span
        # influence is demonstrably present where its field reaches
        assert got["within_span_max_grad"] > 0.05
        # and beyond its own span only the weak GroupNorm statistics
        # coupling remains, under the same bound part 1 pinned for the
        # baseline: the U-Net is not credited with global context
        assert got["beyond_to_within_ratio"] < bound


def test_stage3_flowforge_04_part_2_attention_global_mixing_is_exercised():
    ab, run_id, x, t, regime = _p04_fixture()
    net = BookDDPM(arch="attention", run_id=run_id).net
    convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv1d)]
    attns = [m for m in net.modules()
             if isinstance(m, torch.nn.MultiheadAttention)]
    # the spec is one input convolution, one global mixing layer, one
    # output convolution
    assert len(convs) == 2 and len(attns) == 1
    # every query position places nonzero weight on every key position:
    # the mixing is global, not masked or windowed
    ch = net.in_conv.out_channels
    with torch.no_grad():
        z = torch.randn(2, WINDOW, ch)
        _, weights = net.attn(z, z, z, need_weights=True)
    assert float(weights.min()) > 0.001
    # functional demonstration: perturbing only input offset 0 leaves
    # the attention block's input at the last position exactly unchanged
    # (the input convolution is local), yet the block's output at the
    # last position moves: the attention mix is the only path carrying
    # the perturbation there
    captured = []
    hook = net.attn.register_forward_hook(
        lambda m, i, o: captured.append(o[0].detach().clone()))
    net(x, t, regime)
    xp = x.clone()
    xp[:, :, 0] += 2.0
    net(xp, t, regime)
    hook.remove()
    base, pert = captured
    assert float((pert - base)[:, WINDOW - 1, :].abs().mean()) > 0.001
    with torch.no_grad():
        local_base = net.in_conv(x)
        local_pert = net.in_conv(xp)
    assert float((local_pert - local_base)[:, :, WINDOW - 1]
                 .abs().max()) == 0.0


def test_stage3_flowforge_04_part_2_candidates_within_param_budget():
    ab = _load_ablate()
    baseline = count_params(BookDDPM(run_id=ab.BASELINE_RUN_ID).net)
    assert baseline == 221767
    for arch in ("dilated_tcn", "compact_unet", "attention"):
        n = count_params(BookDDPM(arch=arch,
                                  run_id=ab.BASELINE_RUN_ID).net)
        # the part 1 rule (within the 5 percent ceiling) and the band
        # from the other side: channel width is the knob, and every
        # candidate lands inside the full 5 percent band
        assert ab.within_param_budget(n, baseline)
        assert n >= round(baseline * (1.0 - ab.PARAM_BUDGET_TOLERANCE))


def test_stage3_flowforge_04_part_2_no_shape_or_padding_defects():
    from flowforge.models import DENOISER_ARCHITECTURES
    regime = torch.randn(2, 2)
    for arch, cls in DENOISER_ARCHITECTURES.items():
        net = cls()
        for length in (32, 59, 60, 61, 120):
            x = torch.randn(2, N_FEATURES, length)
            t = torch.zeros(2, dtype=torch.long)
            out = net(x, t, regime)
            assert out.shape == x.shape, (arch, length)
            assert torch.isfinite(out).all(), (arch, length)


def test_stage3_flowforge_04_part_2_bookddpm_integration():
    for arch in ("cnn", "dilated_tcn", "compact_unet", "attention"):
        model = BookDDPM(arch=arch, run_id="p04-part2-smoke",
                         t_steps=10)
        x, regime = _batch(n=4)
        loss = model.p_loss(x.permute(0, 2, 1).contiguous(), regime)
        assert torch.isfinite(loss)
        gen = model.sample(regime, seed=1)
        assert gen.shape == (4, WINDOW, N_FEATURES)
        assert torch.isfinite(gen).all()
    try:
        BookDDPM(arch="transformer")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown architecture must be rejected")


def test_stage3_flowforge_04_part_2_paired_comparison_smoke():
    from flowforge.models import derive_seed, scoped_rng
    ab = _load_ablate()
    with scoped_rng(derive_seed("p04-part2-smoke", "p04:data")):
        splits = {
            name: (torch.randn(n, WINDOW, N_FEATURES),
                   torch.randn(n, 2))
            for name, n in (("train", 48), ("val", 16), ("test", 16))
        }
    record = ab.compare_architectures(
        splits, run_id="p04-part2-smoke", epochs=1, t_steps=20,
        disc_epochs=3, verbose=False)
    # one run, one record: baseline and candidates measured together
    assert record["run_id"] == "p04-part2-smoke"
    assert set(record["architectures"]) == {
        "cnn", "dilated_tcn", "compact_unet", "attention"}
    for arch, entry in record["architectures"].items():
        assert entry["within_param_budget"]
        assert len(entry["overfit_curve"]) == 1
        assert np.isfinite(entry["fidelity_mean_w1"])
        assert 0.0 <= entry["separation_auc_test"] <= 1.0
        assert entry["latency_forward_median_ms_batch32"] > 0
        assert len(entry["empirical_influence_at_last_offset"]
                   ["gradient"]) == len(ab.INFLUENCE_OFFSETS)
