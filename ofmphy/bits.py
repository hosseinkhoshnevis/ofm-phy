"""Bit sources and error counting."""

import numpy as np


def random_bits(n, rng):
    """Uniform payload bits. A seeded Generator keeps runs repeatable;
    swap in a PRBS if you need a hardware-style pattern."""
    return rng.integers(0, 2, size=n, dtype=np.uint8)


def count_errors(a, b):
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: {a.shape} vs {b.shape}")
    return int(np.count_nonzero(a ^ b))


def ber(a, b):
    """(errors, total, ratio). A measured 0 only means < 1/total."""
    n_err = count_errors(a, b)
    return n_err, a.size, n_err / a.size
