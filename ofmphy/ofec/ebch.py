"""Extended BCH(256,239) — the oFEC component code.

Conventions follow the OpenROADM / OpenZR+ text exactly: codeword bit i
(i = 0..254) is the coefficient of y^(254-i), parity sits at positions
239..254 and bit 255 is an overall even-parity extension. The 255-bit
prefix of every valid codeword is divisible by

    g(y) = y^16+y^14+y^13+y^11+y^10+y^9+y^8+y^6+y^5+y+1   (0x16f63)

which factors as m1(y)*m3(y) with m1 = 0x11d (minimal polynomial of alpha,
also the field polynomial) and m3 = 0x177 (of alpha^3). Roots alpha^1..4
give t = 2 and d_min = 5; the extension bit lifts d_min to 6.

Everything below operates on (n_codewords, 256) uint8 arrays.
"""

import numpy as np

from .galois import EXP, LOG, QUAD_ROOT, gf_inv, gf_mul, gf_pow3

G_POLY = 0x16F63
N = 256
K = 239


def _remainders():
    # rem[d] = y^d mod g(y) as a 16-bit mask, bit j <-> coefficient of y^j
    rems = np.zeros(255, dtype=np.int64)
    r = G_POLY & 0xFFFF  # y^16 mod g
    rems[16] = r
    for d in range(17, 255):
        r <<= 1
        if r & 0x10000:
            r ^= G_POLY
        rems[d] = r
    return rems


_REMS = _remainders()


def _parity_matrix():
    # Info position i (0..238) contributes y^(254-i) mod g to the remainder.
    # Parity bit 239+j holds the coefficient of y^(15-j).
    pm = np.zeros((K, 16), dtype=np.uint8)
    for i in range(K):
        r = int(_REMS[254 - i])
        for j in range(16):
            pm[i, j] = (r >> (15 - j)) & 1
    return pm


PARITY_MAT = _parity_matrix()

# Bit i of the 255-bit body has error locator alpha^(254-i)
_pos = np.arange(255)
LOC1 = EXP[(254 - _pos) % 255].copy()
LOC3 = EXP[(3 * ((254 - _pos) % 255)) % 255].copy()


def parity_bits(msg):
    """msg: (..., 239) -> the 16 BCH parity bits, msg layout as in the spec."""
    par = (np.asarray(msg, dtype=np.uint32) @ PARITY_MAT.astype(np.uint32)) & 1
    return par.astype(np.uint8)


def encode(msg):
    """(n, 239) info bits -> (n, 256) codewords (parity + extension appended)."""
    msg = np.atleast_2d(np.asarray(msg, dtype=np.uint8))
    par = parity_bits(msg)
    body = np.concatenate([msg, par], axis=-1)
    ext = np.bitwise_xor.reduce(body, axis=-1, keepdims=True)
    return np.concatenate([body, ext], axis=-1)


def syndromes(cw):
    """S1 = c(alpha), S3 = c(alpha^3), sp = overall parity of all 256 bits."""
    body = np.asarray(cw[:, :255], dtype=np.int32)
    s1 = np.bitwise_xor.reduce(body * LOC1, axis=1)
    s3 = np.bitwise_xor.reduce(body * LOC3, axis=1)
    sp = np.bitwise_xor.reduce(cw.astype(np.int32), axis=1)
    return s1, s3, sp


def bdd_decode(cw):
    """Bounded-distance decode, up to two errors plus the extension bit.

    Returns (corrected, fail). The input array is left untouched; `fail`
    marks words where >= 3 errors were detected (left uncorrected).
    """
    out = np.array(cw, dtype=np.uint8, copy=True)
    s1, s3, sp = syndromes(out)
    s1c = gf_pow3(s1)
    rows = np.arange(out.shape[0])

    # error in the extension bit only
    m = (s1 == 0) & (s3 == 0) & (sp == 1)
    out[m, 255] ^= 1

    # one body error (log of a sanitised copy; masked rows don't care)
    single = (s1 != 0) & (s3 == s1c)
    pos1 = (254 - LOG[np.where(s1 == 0, 1, s1)]) % 255
    m = single & (sp == 1)
    out[rows[m], pos1[m]] ^= 1
    m = single & (sp == 0)          # body error + extension error
    out[rows[m], pos1[m]] ^= 1
    out[m, 255] ^= 1

    fail = (s1 == 0) & (s3 != 0)
    fail |= (s1 != 0) & (s3 != s1c) & (sp == 1)   # three errors detected

    dbl = (s1 != 0) & (s3 != s1c) & (sp == 0)
    if dbl.any():
        s1d = np.where(dbl, s1, 1)
        s3d = np.where(dbl, s3, 0)
        # Locators X1, X2 solve z^2 + S1 z + (S3/S1 + S1^2) = 0.
        # Substituting z = S1 w turns it into w^2 + w = S3/S1^3 + 1.
        c = gf_mul(s3d, gf_inv(gf_pow3(s1d))) ^ 1
        w = QUAD_ROOT[c]
        ok = dbl & (w >= 2)  # w in {0,1} would put a locator at zero; no root -> -1
        wv = np.where(ok, w, 2)
        x1 = gf_mul(s1d, wv)
        x2 = x1 ^ s1d        # S1*(w+1)
        p1 = (254 - LOG[np.where(x1 == 0, 1, x1)]) % 255
        p2 = (254 - LOG[np.where(x2 == 0, 1, x2)]) % 255
        out[rows[ok], p1[ok]] ^= 1
        out[rows[ok], p2[ok]] ^= 1
        fail |= dbl & ~ok
    return out, fail
