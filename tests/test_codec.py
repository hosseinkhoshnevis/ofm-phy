import numpy as np
import pytest

from ofmphy.ofec import OfecCodec


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(42)
    codec = OfecCodec(30)
    payload = rng.integers(0, 2, codec.frame.n_info, dtype=np.uint8)
    coded, matrix = codec.encode(payload)
    return codec, payload, coded, matrix


def test_rate(setup):
    codec, payload, coded, _ = setup
    # steady-state code rate is 111/128; the frame adds leader + termination
    assert codec.frame.n_info / (codec.frame.n_data_rows * 2048) == 111 / 128


def test_hd_roundtrip(setup):
    codec, payload, coded, _ = setup
    assert (codec.decode_hard(coded.copy()) == payload).all()


def test_sd_roundtrip(setup):
    codec, payload, coded, _ = setup
    llr = 6.0 * (1.0 - 2.0 * coded.astype(np.float32))
    assert (codec.decode_soft(llr) == payload).all()


def test_leader_is_deterministic():
    a, b = OfecCodec(4), OfecCodec(4)
    assert (a._leader_bits == b._leader_bits).all()


def test_hd_corrects_below_threshold(setup):
    codec, payload, coded, _ = setup
    rng = np.random.default_rng(1)
    noisy = coded ^ (rng.random(coded.size) < 6e-3).astype(np.uint8)
    assert (codec.decode_hard(noisy) == payload).all()


def test_sd_corrects_near_spec_threshold(setup):
    """AWGN-on-BPSK proxy at pre-FEC BER ~1.8e-2, just under the 2.0e-2
    oFEC operating point. The SD decoder must come back clean."""
    codec, payload, coded, _ = setup
    rng = np.random.default_rng(2)
    sigma = 0.478  # Q(1/sigma) ~ 1.8e-2
    x = 1.0 - 2.0 * coded.astype(np.float32)
    y = x + rng.normal(0.0, sigma, x.size).astype(np.float32)
    pre = float(((y < 0).astype(np.uint8) != coded).mean())
    assert 1.2e-2 < pre < 2.4e-2
    out = codec.decode_soft(2.0 * y / sigma**2)
    assert (out == payload).all()
