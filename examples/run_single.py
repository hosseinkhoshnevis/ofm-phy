#!/usr/bin/env python3
"""Run one frame through the 800G chain and print the scorecard.

    python examples/run_single.py --osnr 23 --rows 60 --decoder both --plots
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ofmphy import SimConfig, run_link
from ofmphy.metrics import theory_ber_16qam


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--osnr", type=float, default=23.0, help="OSNR in 0.1 nm [dB]")
    ap.add_argument("--rows", type=int, default=60, help="payload oFEC block rows")
    ap.add_argument("--decoder", default="both", choices=["sd", "hd", "both"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--plots", action="store_true", help="save constellation PNG")
    args = ap.parse_args()

    cfg = SimConfig(n_data_rows=args.rows, osnr_db=args.osnr,
                    decoder=args.decoder, seed=args.seed)
    r = run_link(cfg, return_symbols=args.plots)

    gbaud = cfg.symbol_rate / 1e9
    print(f"DP-16QAM {gbaud:.6f} GBd, oFEC r=111/128, OSNR {args.osnr:.1f} dB "
          f"(Es/N0 {r['esn0_db']:.1f} dB/pol)")
    print(f"line bits {r['n_line_bits']:,}  payload bits {r['n_info_bits']:,}")
    print(f"pre-FEC  BER {r['pre_fec_ber']:.3e}   ({r['pre_fec_errors']:,} errors, "
          f"theory AWGN {theory_ber_16qam(r['esn0_db']):.3e})")
    for name, p in r["post_fec"].items():
        shown = f"{p['ber']:.3e}" if p["errors"] else f"0 (< {1.0 / p['bits']:.1e})"
        print(f"post-FEC BER {shown}   [{name}, {p['errors']} errors]")
    print(f"EVM {r['evm_pct']:.1f}%   Q {r['q_db']:.2f} dB   "
          f"FOE error {r['freq_offset_err_hz'] / 1e6:+.1f} MHz")
    t = r["time_s"]
    rate = r["n_info_bits"] / t["total"] / 1e3
    print(f"wall time {t['total']:.2f} s (dsp {t['dsp']:.2f}, fec {t['fec']:.2f}) "
          f"-> {rate:.0f} kbit/s simulated")

    if args.plots:
        plot(r, args)


def plot(r, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tx, rx = r["tx_symbols"], r["rx_symbols"]
    n = min(6000, rx.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), facecolor="#fcfcfb")
    for ax, pol, label in zip(axes, (0, 1), ("X", "Y")):
        ax.scatter(rx[pol, :n].real, rx[pol, :n].imag, s=2.2, alpha=0.35,
                   color="#2a78d6", linewidths=0)
        ax.scatter(tx[pol, :n].real, tx[pol, :n].imag, s=9, color="#0b0b0b",
                   marker="+", linewidths=0.8)
        ax.set_title(f"{label} polarisation", fontsize=10, color="#0b0b0b")
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.tick_params(colors="#898781", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#c3c2b7")
        ax.set_facecolor("#fcfcfb")
    fig.suptitle(f"Equalised constellation, OSNR {args.osnr:.1f} dB "
                 f"(EVM {r['evm_pct']:.1f}%)", fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    out = Path(__file__).with_name(f"constellation_{args.osnr:g}dB.png")
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
