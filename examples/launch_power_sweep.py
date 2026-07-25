#!/usr/bin/env python3
"""Q factor against launch power over an amplified nonlinear link, the
classic bell curve. ASE and Kerr move together: OSNR follows the budget
OSNR = P - span_loss + 58 - NF - 10 log10(N_spans), while the split-step
fibre turns launch power into nonlinear phase span after span. Low power
is ASE-limited, high power is Kerr-limited, the optimum sits in between.

Default link: 10 x 80 km (a single ZR-style span never becomes
Kerr-limited at sane powers, which is why it runs error-free everywhere).

    python examples/launch_power_sweep.py --spans 10 --frames 2
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ofmphy import SimConfig, run_link
from ofmphy.channel import osnr_from_launch
from ofmphy.metrics import q_db_from_ber, theory_ber_16qam
from ofmphy.ofec import OfecCodec

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
C_Q = "#2a78d6"
T_OK = "#e4f5ee"  # error-free region wash


def sweep(args):
    codec = OfecCodec(args.rows)
    rows = []
    for p_dbm in np.arange(args.start, args.stop + 1e-9, args.step):
        pre_e = pre_n = sd_e = sd_n = 0
        t0 = time.time()
        for f in range(args.frames):
            cfg = SimConfig(n_data_rows=args.rows, decoder="sd",
                            nl_steps=args.steps, n_spans=args.spans,
                            launch_power_dbm=float(p_dbm),
                            seed=args.seed + f)
            cfg.osnr_db = osnr_from_launch(float(p_dbm), cfg)
            r = run_link(cfg, codec=codec)
            pre_e += r["pre_fec_errors"]
            pre_n += r["n_line_bits"]
            sd_e += r["post_fec"]["sd"]["errors"]
            sd_n += r["post_fec"]["sd"]["bits"]
        pre = pre_e / pre_n
        rows.append({
            "launch_dbm": float(p_dbm),
            "osnr_db": cfg.osnr_db,
            "pre_ber": pre,
            "q_db": q_db_from_ber(pre),
            "ase_only_ber": float(theory_ber_16qam(
                cfg.osnr_db - 10 * np.log10(cfg.symbol_rate / cfg.ref_bw))),
            "sd_errors": sd_e, "sd_bits": sd_n,
        })
        print(f"P {p_dbm:+5.1f} dBm  OSNR {cfg.osnr_db:.1f}  pre {pre:.3e}  "
              f"Q {rows[-1]['q_db']:.2f} dB  sd {sd_e}/{sd_n}  ({time.time()-t0:.1f}s)")
    return rows


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = np.array([r["launch_dbm"] for r in rows])
    q = np.array([r["q_db"] for r in rows])
    q_ase = np.array([q_db_from_ber(r["ase_only_ber"]) for r in rows])
    clean = np.array([r["sd_errors"] == 0 for r in rows])

    fig, ax = plt.subplots(figsize=(7.6, 4.9), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.set_ylim(4.0, 16.0)  # the deep-collapse point exits the frame bottom
    if clean.any():  # wash the region where oFEC comes back error-free
        lo, hi = p[clean].min(), p[clean].max()
        ax.axvspan(lo, hi, color=T_OK, zorder=0)
        ax.text((lo + hi) / 2, 4.45, "oFEC error-free after decoding",
                ha="center", fontsize=8, color="#1baf7a")
    ax.plot(p, q_ase, ls="--", lw=1.2, color=MUTED,
            label="ASE-only bound (theory)")
    ax.plot(p, q, "o-", lw=1.8, ms=4.5, color=C_Q, label="measured, full chain")
    k = int(np.argmax(q))
    ax.annotate(f"optimum {p[k]:+.0f} dBm", (p[k], q[k]),
                textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8.5, color=INK)
    ax.set_xlabel("launch power, both polarisations [dBm]", color=INK)
    ax.set_ylabel("Q from pre-FEC BER [dB]", color=INK)
    ax.set_title("10 x 80 km amplified link, split-step Manakov fibre\n"
                 "OSNR tied to launch power: OSNR = P - 16 + 58 - NF(5) - 10log10(10)",
                 fontsize=10.5, color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(BASE)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=float, default=-4.0)
    ap.add_argument("--stop", type=float, default=12.0)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--spans", type=int, default=10, help="80 km spans")
    ap.add_argument("--steps", type=int, default=60, help="SSFM steps per span")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--outdir", default=str(Path(__file__).parent))
    args = ap.parse_args()

    rows = sweep(args)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "launch_power_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, out / "launch_power_sweep.png")


if __name__ == "__main__":
    main()
