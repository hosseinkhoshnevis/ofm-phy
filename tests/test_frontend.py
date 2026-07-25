"""Front-end imperfections and their compensation blocks."""

import numpy as np
import pytest

from ofmphy import SimConfig, channel, run_link, rxdsp

FS = 236.4e9


def proper_signal(n=65536, seed=0):
    """Bandlimited circularly-symmetric test signal, two polarisations."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2, n)) + 1j * rng.normal(size=(2, n))
    F = np.fft.fft(x, axis=1)
    f = np.fft.fftfreq(n, 1.0 / FS)
    F[:, np.abs(f) > 62e9] = 0.0  # keep it inside a 16QAM-ish bandwidth
    return np.fft.ifft(F, axis=1)


def improperness(x):
    """|E[x^2]| / E[|x|^2] - zero for a clean complex baseband signal."""
    return abs(np.mean(x ** 2)) / np.mean(np.abs(x) ** 2)


def test_imbalance_creates_image_and_gsop_removes_it():
    x = proper_signal()
    bad = channel.apply_iq_imperfections(x, FS, 0.5, 2.0, 0.0)
    assert improperness(bad[0]) > 0.02
    fixed = rxdsp.frontend_correction(bad, FS, 0.0)
    assert improperness(fixed[0]) < 1e-3
    assert improperness(fixed[1]) < 1e-3


def test_deskew_inverts_skew():
    x = proper_signal()
    bad = channel.apply_iq_imperfections(x, FS, 0.0, 0.0, 1.5)
    fixed = rxdsp.frontend_correction(bad, FS, 1.5)
    err = np.mean(np.abs(fixed - x) ** 2) / np.mean(np.abs(x) ** 2)
    assert err < 1e-3  # GSOP renormalisation leaves a whisker, nothing more


def test_adc_quantiser_noise_floor():
    x = proper_signal()
    q = channel.adc_quantise(x, 6)
    snr = np.mean(np.abs(x) ** 2) / np.mean(np.abs(q - x) ** 2)
    assert 25.0 < 10 * np.log10(snr) < 45.0  # ~6 ENOB with 3.2 sigma clipping


def test_wl_correct_removes_conjugate_mixing():
    rng = np.random.default_rng(3)
    from ofmphy import mapping
    s = mapping.bits_to_symbols(rng.integers(0, 2, 8 * 20000, dtype=np.uint8))
    z = 0.98 * s + 0.06 * np.exp(0.4j) * np.conj(s)
    z = z + 0.02 * (rng.normal(size=s.shape) + 1j * rng.normal(size=s.shape))
    out = rxdsp.wl_correct(z, s, n_taps=3, block=4096)
    evm_in = np.sqrt(np.mean(np.abs(z - s) ** 2))
    evm_out = np.sqrt(np.mean(np.abs(out - s) ** 2))
    assert evm_out < 0.5 * evm_in


@pytest.mark.parametrize("comp", [True, False])
def test_chain_with_frontend_impairments(comp):
    """The compensated chain has to land near the clean baseline; the
    uncompensated one has to be visibly broken - otherwise these blocks
    would be decorative."""
    kw = {} if comp else dict(frontend_comp=False, eq_widely_linear=False, wl_taps=0)
    r = run_link(SimConfig(n_data_rows=30, osnr_db=35.0, decoder="hd",
                           hd_iters=1, **kw))
    if comp:
        assert r["evm_pct"] < 9.0
    else:
        assert r["evm_pct"] > 12.0


def test_dd_lms_does_not_hurt_and_helps_near_threshold():
    """The decision-directed pass must never make things worse, and near the
    FEC operating point it has real ISI left to clean up."""
    base = dict(n_data_rows=40, osnr_db=23.5, decoder="hd", hd_iters=1)
    rde = run_link(SimConfig(**base, dd_sweeps=0))
    dd = run_link(SimConfig(**base))
    assert dd["evm_pct"] < rde["evm_pct"] + 0.05
    assert dd["pre_fec_ber"] < rde["pre_fec_ber"] * 0.95


def test_ml_phase_refine_removes_static_offset():
    rng = np.random.default_rng(9)
    from ofmphy import mapping
    s = mapping.bits_to_symbols(rng.integers(0, 2, 8 * 8000, dtype=np.uint8))
    noisy = s + 0.05 * (rng.normal(size=s.shape) + 1j * rng.normal(size=s.shape))
    rot = noisy * np.exp(1j * 0.06)  # inside the slicing basin, sub-BPS-grid
    out = rxdsp.ml_phase_refine(rot, window=160, iters=2)
    assert np.mean(np.abs(out - s) ** 2) < 1.05 * np.mean(np.abs(noisy - s) ** 2)


def test_llrs_accept_per_polarisation_variance():
    from ofmphy import mapping
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, 8 * 3000, dtype=np.uint8)
    sym = mapping.bits_to_symbols(bits)
    l = mapping.llrs(sym + 1e-3, np.array([0.01, 0.04]))
    per_sym = np.abs(l.reshape(-1, 8))
    # cleaner polarisation (X, first 4 bits) must carry larger magnitudes
    assert per_sym[:, :4].mean() > 2.5 * per_sym[:, 4:].mean()
    assert ((l < 0).astype(np.uint8) == bits).all()


def test_compensated_matches_clean_baseline():
    clean = dict(tx_iq_amp_db=0, tx_iq_phase_deg=0, tx_iq_skew_ps=0,
                 rx_iq_amp_db=0, rx_iq_phase_deg=0, rx_iq_skew_ps=0,
                 adc_bits=0, pdl_db=0)
    base = run_link(SimConfig(n_data_rows=30, osnr_db=35.0, decoder="hd",
                              hd_iters=1, **clean))
    full = run_link(SimConfig(n_data_rows=30, osnr_db=35.0, decoder="hd",
                              hd_iters=1))
    assert full["evm_pct"] < base["evm_pct"] + 2.0
