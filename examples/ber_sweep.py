#!/usr/bin/env python3
"""OSNR waterfall: pre- and post-FEC BER across the operating range.

    python examples/ber_sweep.py --start 21.5 --stop 27 --step 0.5 --frames 3

Writes a CSV next to this file and a waterfall plot. Zero-error points are
drawn as open markers at the measurement floor (1/bits counted).
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ofmphy import SimConfig, run_link
from ofmphy.metrics import theory_ber_16qam
from ofmphy.ofec import OfecCodec

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
C_PRE = "#2a78d6"   # measured pre-FEC
C_SD = "#eb6834"    # soft-decision post-FEC
C_HD = "#1baf7a"    # hard-decision post-FEC


def sweep(args):
    codec = OfecCodec(args.rows)
    if args.points:
        osnrs = [float(x) for x in args.points.split(",")]
    else:
        osnrs = list(np.arange(args.start, args.stop + 1e-9, args.step))
    rows = []
    for osnr in osnrs:
        pre_e = pre_n = 0
        post = {"sd": [0, 0], "hd": [0, 0]}
        t0 = time.time()
        f = 0
        # In the decoder transition region, keep adding frames until the
        # soft-decision error count is statistically meaningful or the
        # frame budget runs out. The steep part of the waterfall needs
        # far more bits than the flanks.
        while f < args.frames or (args.min_errors
                                  and post["sd"][0] < args.min_errors
                                  and f < args.max_frames):
            cfg = SimConfig(n_data_rows=args.rows, osnr_db=float(osnr),
                            decoder="both", seed=args.seed + f)
            r = run_link(cfg, codec=codec)
            pre_e += r["pre_fec_errors"]
            pre_n += r["n_line_bits"]
            for k, p in r["post_fec"].items():
                post[k][0] += p["errors"]
                post[k][1] += p["bits"]
            f += 1
        row = {
            "osnr_db": float(osnr),
            "esn0_db": r["esn0_db"],
            "pre_ber": pre_e / pre_n,
            "sd_errors": post["sd"][0], "sd_bits": post["sd"][1],
            "hd_errors": post["hd"][0], "hd_bits": post["hd"][1],
            "theory_ber": float(theory_ber_16qam(r["esn0_db"])),
        }
        rows.append(row)
        print(f"OSNR {osnr:5.2f}: pre {row['pre_ber']:.3e}  "
              f"sd {post['sd'][0]:6d}/{post['sd'][1]}  "
              f"hd {post['hd'][0]:6d}/{post['hd'][1]}  "
              f"({f} frames, {time.time() - t0:.1f}s)")
    return rows


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {path}")


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    o = np.array([r["osnr_db"] for r in rows])
    pre = np.array([r["pre_ber"] for r in rows])
    th = np.array([r["theory_ber"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.6, 5.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(o, th, ls="--", lw=1.2, color=MUTED, label="16QAM AWGN theory")
    ax.plot(o, pre, "o-", lw=1.8, ms=4.5, color=C_PRE, label="pre-FEC, measured")

    for key, col, label, dodge in (("sd", C_SD, "post-FEC oFEC soft", 1.0),
                                   ("hd", C_HD, "post-FEC oFEC hard", 1.45)):
        x_hit, y_hit, x_zero, y_floor = [], [], [], []
        for r in rows:
            e, n = r[f"{key}_errors"], r[f"{key}_bits"]
            if e:
                x_hit.append(r["osnr_db"]); y_hit.append(e / n)
            else:
                x_zero.append(r["osnr_db"]); y_floor.append(dodge / n)
        ax.plot(x_hit, y_hit, "s-", lw=1.8, ms=4.5, color=col, label=label)
        if x_zero:  # zero errors: open marker at the counting floor (dodged so
            ax.plot(x_zero, y_floor, "v", ms=6, mfc="none", mec=col, mew=1.4)
            # the two decoders' floors don't sit on top of each other)

    ax.axhline(2.0e-2, color=BASE, lw=1.0, ls=":")
    ax.text(o[-1], 2.35e-2, "oFEC threshold 2.0e-2", ha="right",
            fontsize=8, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(3e-7, 2e-1)
    ax.set_xlabel("OSNR in 0.1 nm [dB]", color=INK)
    ax.set_ylabel("bit error rate", color=INK)
    ax.set_title("DP-16QAM 118.2 GBd with OpenROADM oFEC\n"
                 "open markers = zero errors observed (counting floor)",
                 fontsize=10.5, color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(BASE)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=float, default=21.5)
    ap.add_argument("--stop", type=float, default=27.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--frames", type=int, default=3, help="frames per OSNR point")
    ap.add_argument("--rows", type=int, default=100, help="payload block rows/frame")
    ap.add_argument("--points", default="",
                    help="comma list of OSNR points, replacing the start/stop grid")
    ap.add_argument("--min-errors", type=int, default=0,
                    help="in the transition region, add frames until this many "
                         "soft-decision errors have been counted")
    ap.add_argument("--max-frames", type=int, default=30,
                    help="frame budget per point when --min-errors is set")
    ap.add_argument("--merge", action="store_true",
                    help="merge with the existing CSV instead of overwriting")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--outdir", default=str(Path(__file__).parent))
    args = ap.parse_args()

    rows = sweep(args)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ber_sweep.csv"
    if args.merge and path.exists():
        seen = {r["osnr_db"] for r in rows}
        with open(path) as f:
            for old in csv.DictReader(f):
                if float(old["osnr_db"]) not in seen:
                    rows.append({k: (int(v) if k.endswith(("errors", "bits"))
                                     else float(v)) for k, v in old.items()})
        rows.sort(key=lambda r: r["osnr_db"])
    save_csv(rows, path)
    plot(rows, out / "ber_sweep.png")


if __name__ == "__main__":
    main()
