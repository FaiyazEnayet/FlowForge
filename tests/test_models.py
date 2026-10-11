"""Model shape tests: forward passes and generation produce sane tensors."""

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
