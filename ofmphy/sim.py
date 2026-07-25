"""End-to-end link runner: bits -> oFEC -> DP-16QAM -> channel -> DSP -> BER."""

import time

import numpy as np

from . import bits, channel, mapping, rxdsp, shaping
from .config import SimConfig
from .metrics import evm_percent, q_db_from_ber
from .ofec import OfecCodec

_SCRAMBLER_SEED = 0x5C2A


def _scrambler(n):
    """Fixed line scrambler, same at both ends. The oFEC termination and
    leader regions are heavily structured (long zero runs map to outer
    constellation corners), and every stage from the CMA radii to the BPS
    decisions assumes white-ish symbols, which is exactly why real framers
    scramble the line."""
    return np.random.default_rng(_SCRAMBLER_SEED).integers(0, 2, n, dtype=np.uint8)


def run_link(cfg=None, seed=None, codec=None, return_symbols=False):
    """Run one frame through the whole chain.

    Pass a prebuilt codec when sweeping; the index maps are frame-size
    dependent and worth reusing. Returns a dict of measurements.
    """
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    codec = codec or OfecCodec(cfg.n_data_rows)
    t0 = time.perf_counter()

    # ---------------- transmit
    payload = bits.random_bits(codec.frame.n_info, rng)
    coded, _ = codec.encode(payload)
    scr = _scrambler(coded.size)
    tx_sym = mapping.bits_to_symbols(coded ^ scr)
    wave = shaping.shape(tx_sym, cfg.sps, cfg.symbol_rate, cfg.rolloff)
    t1 = time.perf_counter()

    # ---------------- channel
    rx_wave = channel.apply_channel(wave, cfg, rng)
    t2 = time.perf_counter()

    # ---------------- receiver DSP
    s = rx_wave
    if cfg.frontend_comp:
        s = rxdsp.frontend_correction(s, cfg.sample_rate, cfg.rx_iq_skew_ps)
    s = rxdsp.cd_compensate(s, cfg)
    s = shaping.matched_filter(s, cfg.sps, cfg.symbol_rate, cfg.rolloff)
    fo_hat = rxdsp.estimate_freq_offset(s, cfg.sample_rate)
    s = rxdsp.remove_freq_offset(s, cfg.sample_rate, fo_hat)
    eq = rxdsp.ButterflyEqualizer(cfg.eq_taps, cfg.sps, cfg.eq_block,
                                  widely_linear=cfg.eq_widely_linear)
    y = eq.run(s, cfg.cma_sweeps, cfg.rde_sweeps, cfg.mu_cma, cfg.mu_rde)
    y, track = rxdsp.blind_phase_search(y, cfg.bps_phases, cfg.bps_window)
    if cfg.dd_sweeps:
        # decision-directed LMS against the recovered carrier, then a fresh
        # phase estimate on the refined output
        y = eq.refine_dd(track, cfg.mu_dd, cfg.dd_sweeps)
        y, _ = rxdsp.blind_phase_search(y, cfg.bps_phases, cfg.bps_window)
    if cfg.ml_window:
        y = rxdsp.ml_phase_refine(y, cfg.ml_window, cfg.ml_iters)
    y, noise_var = rxdsp.align_to_reference(y, tx_sym)
    if cfg.wl_taps:
        y = rxdsp.wl_correct(y, tx_sym, cfg.wl_taps, cfg.wl_block)
    # per-polarisation calibration: with PDL the two pols see different SNR
    nv_pol = np.mean(np.abs(y - tx_sym) ** 2, axis=1)
    noise_var = float(nv_pol.mean())
    t3 = time.perf_counter()

    # ---------------- measure + decode
    hard = mapping.symbols_to_bits(y) ^ scr
    pre_err, n_line, pre_ber = bits.ber(hard, coded)
    llr = mapping.llrs(y, nv_pol) * (1.0 - 2.0 * scr)  # descramble = flip signs
    post = {}
    if cfg.decoder in ("sd", "both"):
        out = codec.decode_soft(llr, n_iter=cfg.sd_iters, n_lrb=cfg.chase_lrb,
                                alphas=cfg.alphas, betas=cfg.betas,
                                hd_cleanup=cfg.hd_cleanup)
        post["sd"] = bits.ber(out, payload)
    if cfg.decoder in ("hd", "both"):
        out = codec.decode_hard(hard, n_iter=cfg.hd_iters)
        post["hd"] = bits.ber(out, payload)
    t4 = time.perf_counter()

    res = {
        "osnr_db": cfg.osnr_db,
        "esn0_db": channel.esn0_db_from_osnr(cfg.osnr_db, cfg.symbol_rate, cfg.ref_bw),
        "pre_fec_ber": pre_ber,
        "pre_fec_errors": pre_err,
        "n_line_bits": n_line,
        "post_fec": {k: {"errors": v[0], "bits": v[1], "ber": v[2]} for k, v in post.items()},
        "n_info_bits": payload.size,
        "q_db": q_db_from_ber(pre_ber),
        "evm_pct": evm_percent(y, tx_sym),
        "noise_var": noise_var,
        "freq_offset_est_hz": fo_hat,
        "freq_offset_err_hz": fo_hat - cfg.freq_offset_hz,
        "time_s": {"tx": t1 - t0, "channel": t2 - t1, "dsp": t3 - t2,
                   "fec": t4 - t3, "total": t4 - t0},
    }
    if return_symbols:
        res["tx_symbols"] = tx_sym
        res["rx_symbols"] = y
    return res
