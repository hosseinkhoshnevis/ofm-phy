import numpy as np
import pytest

from ofmphy import channel, mapping, rxdsp, shaping
from ofmphy.config import SimConfig


@pytest.fixture()
def tx():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 8 * 16384, dtype=np.uint8)
    sym = mapping.bits_to_symbols(bits)
    return sym, shaping.shape(sym, 2, 118.2e9, 0.05)


def test_shaping_is_isi_free(tx):
    sym, wave = tx
    m = shaping.matched_filter(wave, 2, 118.2e9, 0.05)
    pick = m[:, ::2]
    g = np.vdot(pick.ravel(), sym.ravel()) / np.vdot(sym.ravel(), sym.ravel())
    evm = np.sqrt(np.mean(np.abs(pick - g * sym) ** 2) / np.mean(np.abs(g * sym) ** 2))
    assert evm < 0.02


def test_cd_roundtrip_exact(tx):
    _, wave = tx
    cfg = SimConfig()
    fs = cfg.sample_rate
    out = channel.apply_cd(channel.apply_cd(wave, fs, cfg), fs, cfg, invert=True)
    assert np.abs(out - wave).max() < 1e-9


def test_osnr_calibration(tx):
    _, wave = tx
    cfg = SimConfig(osnr_db=20.0)
    rng = np.random.default_rng(1)
    noisy = channel.apply_ase(wave, cfg, rng)
    var = np.mean(np.abs(noisy - wave) ** 2)
    want = channel.osnr_to_noise_var(20.0, cfg.symbol_rate, cfg.sps, cfg.ref_bw, 1.0)
    assert abs(var / want - 1.0) < 0.05


def test_foe_accuracy(tx):
    _, wave = tx
    cfg = SimConfig(osnr_db=24.0, fibre_km=0, dgd_ps=0, pol_theta1=0, pol_theta2=0,
                    linewidth_hz=0, freq_offset_hz=0.8e9)
    rng = np.random.default_rng(2)
    rx = channel.apply_channel(wave, cfg, rng)
    m = shaping.matched_filter(rx, cfg.sps, cfg.symbol_rate, cfg.rolloff)
    est = rxdsp.estimate_freq_offset(m, cfg.sample_rate)
    assert abs(est - 0.8e9) < 20e6


def test_polarisation_is_unitary_without_pdl(tx):
    _, wave = tx
    cfg = SimConfig(pdl_db=0)
    out = channel.apply_polarisation(wave, cfg.sample_rate, cfg)
    p_in = np.sum(np.abs(wave) ** 2)
    p_out = np.sum(np.abs(out) ** 2)
    assert abs(p_out / p_in - 1.0) < 1e-9
