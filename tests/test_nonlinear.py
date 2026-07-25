"""Split-step fibre model sanity checks."""

import numpy as np
import pytest

from ofmphy import SimConfig, channel, mapping, run_link, shaping


@pytest.fixture()
def wave():
    rng = np.random.default_rng(0)
    sym = mapping.bits_to_symbols(rng.integers(0, 2, 8 * 16384, dtype=np.uint8))
    return shaping.shape(sym, 2, 118.2e9, 0.05)


def test_zero_gamma_reduces_to_linear_cd(wave):
    cfg = SimConfig(nl_steps=64, gamma_w_km=0.0)
    out_nl = channel.ssfm_propagate(wave, cfg)
    out_lin = channel.apply_cd(wave, cfg.sample_rate, cfg)
    err = np.abs(out_nl - out_lin).max() / np.abs(out_lin).max()
    assert err < 1e-9  # loss and EDFA gain cancel exactly, CD splits cleanly


def test_power_is_restored_after_span(wave):
    cfg = SimConfig(nl_steps=64, launch_power_dbm=3.0)
    out = channel.ssfm_propagate(wave, cfg)
    ratio = np.mean(np.abs(out) ** 2) / np.mean(np.abs(wave) ** 2)
    assert abs(ratio - 1.0) < 1e-6  # Kerr is a pure phase rotation


def test_low_power_matches_linear_link():
    """At -10 dBm over one span the nonlinear phase is a few mrad; the full
    chain must land on the linear result."""
    kw = dict(n_data_rows=30, osnr_db=27.0, decoder="hd", hd_iters=1)
    lin = run_link(SimConfig(**kw))
    nl = run_link(SimConfig(**kw, nl_steps=100, launch_power_dbm=-10.0))
    assert abs(nl["evm_pct"] - lin["evm_pct"]) < 0.5


def test_high_power_hurts():
    kw = dict(n_data_rows=30, osnr_db=27.0, decoder="hd", hd_iters=1)
    lo = run_link(SimConfig(**kw, nl_steps=100, launch_power_dbm=0.0))
    hi = run_link(SimConfig(**kw, nl_steps=100, launch_power_dbm=14.0))
    assert hi["evm_pct"] > lo["evm_pct"] + 2.0


def test_osnr_budget():
    cfg = SimConfig()  # 80 km at 0.2 dB/km, NF 5
    assert abs(channel.osnr_from_launch(2.0, cfg) - (2.0 - 16.0 + 58.0 - 5.0)) < 1e-9
