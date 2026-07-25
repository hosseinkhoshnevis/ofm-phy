"""Link quality metrics."""

import math

import numpy as np


def qfunc(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def q_db_from_ber(b):
    """Gaussian Q-factor equivalent of a BER, in dB (20 log10)."""
    if b <= 0:
        return float("inf")
    if b >= 0.5:
        return -float("inf")
    lo, hi = 0.0, 40.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if qfunc(mid) > b:
            lo = mid
        else:
            hi = mid
    q = 0.5 * (lo + hi)
    return 20.0 * math.log10(q) if q > 0 else -float("inf")


def evm_percent(y, ref):
    """RMS error vector magnitude relative to the reference constellation."""
    err = np.mean(np.abs(y - ref) ** 2)
    return 100.0 * math.sqrt(err / np.mean(np.abs(ref) ** 2))


def theory_ber_16qam(esn0_db):
    """Exact Gray-mapped 16QAM bit error rate on AWGN (per polarisation).

    Derived per PAM4 rail: Pb = (3 Q1 + 2 Q3 - Q5)/4 with Qk = Q(k*sqrt(Es/(5 N0))).
    """
    out = []
    for v in np.atleast_1d(esn0_db):
        x = math.sqrt(10.0 ** (v / 10.0) / 5.0)
        out.append((3 * qfunc(x) + 2 * qfunc(3 * x) - qfunc(5 * x)) / 4.0)
    return np.array(out) if np.ndim(esn0_db) else out[0]
