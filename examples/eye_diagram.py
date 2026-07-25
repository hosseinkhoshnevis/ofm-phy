#!/usr/bin/env python3
"""Eye diagrams around the equalizer: the I rail of the X polarisation,
reconstructed at 8 samples per UI.

Left panel: after CD compensation, matched filtering and frequency offset
removal but before the equalizer, where polarisation mixing, DGD, phase
noise and the IQ front ends keep the eye shut. Right panel: the fully
recovered symbol stream (CMA/RDE/DD-LMS equalizer, BPS, alignment, WL
fit) pushed back through the Nyquist pulse, giving the classic open
4-level 16QAM eye whose fuzz is the residual EVM.

    python examples/eye_diagram.py --osnr 26
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ofmphy import SimConfig
from ofmphy import bits as B
from ofmphy import channel, mapping, rxdsp, shaping
from ofmphy.ofec import OfecCodec
from ofmphy.sim import _scrambler

INK = "#0b0b0b"
MUTED = "#898781"
BASE = "#c3c2b7"
TRACE = "#2a78d6"


def resample_fft(x, up):
    """Bandlimited upsampling by zero-padding the spectrum (circular)."""
    n = x.size
    F = np.fft.fft(x)
    G = np.zeros(n * up, dtype=complex)
    h = n // 2
    G[:h] = F[:h]
    G[-h:] = F[-h:]
    return np.fft.ifft(G) * up


def capture(cfg):
    """Run the chain manually, tap the signal before and after the EQ."""
    rng = np.random.default_rng(cfg.seed)
    codec = OfecCodec(cfg.n_data_rows)
    coded, _ = codec.encode(B.random_bits(codec.frame.n_info, rng))
    tx = mapping.bits_to_symbols(coded ^ _scrambler(coded.size))
    wave = shaping.shape(tx, cfg.sps, cfg.symbol_rate, cfg.rolloff)
    rxw = channel.apply_channel(wave, cfg, rng)

    s = rxdsp.frontend_correction(rxw, cfg.sample_rate, cfg.rx_iq_skew_ps)
    s = rxdsp.cd_compensate(s, cfg)
    s = shaping.matched_filter(s, cfg.sps, cfg.symbol_rate, cfg.rolloff)
    fo = rxdsp.estimate_freq_offset(s, cfg.sample_rate)
    s = rxdsp.remove_freq_offset(s, cfg.sample_rate, fo)
    pre = s[0] / np.sqrt(np.mean(np.abs(s[0]) ** 2))  # X pol, 2 sps

    eq = rxdsp.ButterflyEqualizer(cfg.eq_taps, cfg.sps, cfg.eq_block,
                                  widely_linear=cfg.eq_widely_linear)
    y = eq.run(s, cfg.cma_sweeps, cfg.rde_sweeps, cfg.mu_cma, cfg.mu_rde)
    y, track = rxdsp.blind_phase_search(y, cfg.bps_phases, cfg.bps_window)
    y = eq.refine_dd(track, cfg.mu_dd, cfg.dd_sweeps)
    y, _ = rxdsp.blind_phase_search(y, cfg.bps_phases, cfg.bps_window)
    y, _ = rxdsp.align_to_reference(y, tx)
    y = rxdsp.wl_correct(y, tx, cfg.wl_taps, cfg.wl_block)

    # reconstruct the recovered stream through the raised-cosine pulse
    up = np.zeros(y.shape[1] * 2, dtype=complex)
    up[::2] = y[0]
    H = shaping.rrc_response(up.size, cfg.symbol_rate * 2, cfg.symbol_rate,
                             cfg.rolloff) ** 2
    post = np.fft.ifft(np.fft.fft(up) * H) * 2
    post = post / np.sqrt(np.mean(np.abs(post) ** 2))
    return pre, post  # both at 2 samples per UI


def eye_traces(sig2sps, n_traces, up=4):
    """(n,) at 2 sps -> (traces, 2 UI at 8 sps), symbol centred."""
    x8 = resample_fft(sig2sps, up)  # 8 samples per UI
    spu = 2 * up
    n_ui = x8.size // spu - 2
    take = np.linspace(0, n_ui - 1, min(n_traces, n_ui)).astype(int)
    idx = take[:, None] * spu + np.arange(2 * spu + 1)[None, :]
    return x8.real[idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--osnr", type=float, default=26.0)
    ap.add_argument("--rows", type=int, default=30)
    ap.add_argument("--traces", type=int, default=1400)
    ap.add_argument("--outdir", default=str(Path(__file__).parent))
    args = ap.parse_args()

    cfg = SimConfig(n_data_rows=args.rows, osnr_db=args.osnr, decoder="hd")
    pre, post = capture(cfg)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), facecolor="#fcfcfb",
                             sharey=True)
    t = np.linspace(-1.0, 1.0, 2 * 8 + 1)
    for ax, sig, title in (
            (axes[0], pre, "after CD comp + FOE, before equalizer"),
            (axes[1], post, "after full DSP (through the Nyquist pulse)")):
        tr = eye_traces(sig, args.traces)
        ax.plot(t, tr.T, color=TRACE, lw=0.5, alpha=0.055)
        ax.set_facecolor("#fcfcfb")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.55, 1.55)
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_xlabel("time [UI]", color=MUTED, fontsize=8.5)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(BASE)
    for lev in (-3, -1, 1, 3):
        axes[1].axhline(lev / np.sqrt(10), color=BASE, lw=0.6, ls=":", zorder=0)
    axes[0].set_ylabel("I rail, X polarisation", color=MUTED, fontsize=8.5)
    fig.suptitle(f"16QAM eye at 118.2 GBd, OSNR {args.osnr:g} dB, "
                 "8 samples/UI", fontsize=10.5, color=INK)
    fig.tight_layout()
    out = Path(args.outdir) / "eye_diagram.png"
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
