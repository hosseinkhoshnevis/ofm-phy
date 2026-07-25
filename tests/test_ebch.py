import numpy as np
import pytest

from ofmphy.ofec import ebch
from ofmphy.ofec.galois import EXP, LOG, QUAD_ROOT, gf_mul


def test_field_tables():
    # alpha has order 255 and the log/exp tables invert each other
    assert EXP[0] == 1 and EXP[255] == 1
    for a in (1, 2, 37, 254, 255):
        assert EXP[LOG[a]] == a
    # quadratic table really solves w^2 + w = c
    for c in range(256):
        w = QUAD_ROOT[c]
        if w >= 0:
            assert (int(gf_mul(w, w)) ^ w) == c


def test_generator_polynomial():
    # g = m1 * m3 as claimed in the spec commentary
    def polymul(a, b):
        out = 0
        while b:
            if b & 1:
                out ^= a
            a <<= 1
            b >>= 1
        return out

    assert polymul(0x11D, 0x177) == ebch.G_POLY


def test_encode_is_valid():
    rng = np.random.default_rng(0)
    cw = ebch.encode(rng.integers(0, 2, (300, 239), dtype=np.uint8))
    s1, s3, sp = ebch.syndromes(cw)
    assert not s1.any() and not s3.any() and not sp.any()


@pytest.mark.parametrize("n_err", [1, 2])
def test_corrects_up_to_two_errors(n_err):
    rng = np.random.default_rng(n_err)
    cw = ebch.encode(rng.integers(0, 2, (500, 239), dtype=np.uint8))
    noisy = cw.copy()
    for i in range(noisy.shape[0]):
        locs = rng.choice(256, size=n_err, replace=False)  # extension bit included
        noisy[i, locs] ^= 1
    dec, fail = ebch.bdd_decode(noisy)
    assert not fail.any()
    assert (dec == cw).all()


def test_three_errors_flagged_or_miscorrected_never_silent():
    rng = np.random.default_rng(3)
    cw = ebch.encode(rng.integers(0, 2, (500, 239), dtype=np.uint8))
    noisy = cw.copy()
    for i in range(noisy.shape[0]):
        locs = rng.choice(255, size=3, replace=False)
        noisy[i, locs] ^= 1
    dec, fail = ebch.bdd_decode(noisy)
    # d_min = 6, so 3 errors are always detectable: no decode may claim success
    # while silently returning a wrong codeword at distance <= 2 from the input
    assert fail.all()
