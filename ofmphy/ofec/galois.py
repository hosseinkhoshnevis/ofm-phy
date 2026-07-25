"""GF(2^8) arithmetic tables for the oFEC component code.

The field is GF(256) over the primitive polynomial x^8+x^4+x^3+x^2+1 (0x11d),
the same field as the G.709 RS(255,239). Everything is table-driven so the
BCH decoder can process whole arrays of codewords at once.
"""

import numpy as np

PRIM_POLY = 0x11D


def _build_tables():
    exp = np.zeros(510, dtype=np.int32)
    log = np.zeros(256, dtype=np.int32)
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= PRIM_POLY
    exp[255:] = exp[:255]  # spare period so exponent sums never need a modulo
    return exp, log


EXP, LOG = _build_tables()


def gf_mul(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    out = EXP[(LOG[a] + LOG[b]) % 255]
    return np.where((a == 0) | (b == 0), 0, out)


def gf_inv(a):
    """1/a elementwise; caller guarantees a != 0."""
    a = np.asarray(a)
    return EXP[(255 - LOG[a]) % 255]


def gf_pow3(a):
    a = np.asarray(a)
    return np.where(a == 0, 0, EXP[(3 * LOG[a]) % 255])


def _build_quad_table():
    # One root of w^2 + w = c for each solvable c (Tr(c) = 0, half the field).
    # The other root is w ^ 1. Unsolvable entries stay -1.
    root = np.full(256, -1, dtype=np.int32)
    for w in range(256):
        c = int(gf_mul(w, w)) ^ w
        if root[c] < 0:
            root[c] = w
    return root


QUAD_ROOT = _build_quad_table()
