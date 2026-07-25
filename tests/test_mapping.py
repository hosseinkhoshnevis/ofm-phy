import numpy as np

from ofmphy import mapping


def test_bit_symbol_roundtrip():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 8 * 5000, dtype=np.uint8)
    sym = mapping.bits_to_symbols(bits)
    assert (mapping.symbols_to_bits(sym) == bits).all()


def test_unit_energy():
    rng = np.random.default_rng(1)
    sym = mapping.bits_to_symbols(rng.integers(0, 2, 8 * 200000, dtype=np.uint8))
    assert abs(np.mean(np.abs(sym) ** 2) - 1.0) < 0.01


def test_nearest_symbols_is_idempotent():
    rng = np.random.default_rng(2)
    sym = mapping.bits_to_symbols(rng.integers(0, 2, 8 * 1000, dtype=np.uint8))
    noisy = sym + 0.05 * (rng.normal(size=sym.shape) + 1j * rng.normal(size=sym.shape))
    dec = mapping.nearest_symbols(noisy)
    assert np.allclose(mapping.nearest_symbols(dec), dec)
    assert np.allclose(dec, sym)


def test_llr_signs_match_bits():
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, 8 * 4000, dtype=np.uint8)
    sym = mapping.bits_to_symbols(bits)
    noisy = sym + 0.02 * (rng.normal(size=sym.shape) + 1j * rng.normal(size=sym.shape))
    llr = mapping.llrs(noisy, 2 * 0.02**2)
    assert ((llr < 0).astype(np.uint8) == bits).all()


def test_gray_neighbours_differ_by_one_bit():
    # adjacent PAM4 levels must differ in exactly one bit on each rail
    levels = [-3.0, -1.0, 1.0, 3.0]
    def bits_of(level):
        idx = mapping.GRAY_LEVELS.tolist().index(level)
        return (idx >> 1, idx & 1)
    for a, b in zip(levels[:-1], levels[1:]):
        ba, bb = bits_of(a), bits_of(b)
        assert sum(x != y for x, y in zip(ba, bb)) == 1
