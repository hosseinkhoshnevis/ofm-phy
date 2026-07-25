"""Pulse shaping. Root-raised-cosine applied in the frequency domain.

The whole frame is treated as circular, which buys three nice things for a
simulation: no filter edge transients, chromatic dispersion inverts
exactly, and the delay bookkeeping stays trivial. The frame is long enough
(tens of thousands of symbols) that the wrap-around is statistically
irrelevant.
"""

import numpy as np


def rrc_response(nfft, fs, symbol_rate, rolloff):
    """Sampled sqrt-raised-cosine magnitude response on the fft grid."""
    f = np.abs(np.fft.fftfreq(nfft, 1.0 / fs))
    T = 1.0 / symbol_rate
    f1 = (1.0 - rolloff) / (2.0 * T)
    f2 = (1.0 + rolloff) / (2.0 * T)
    H = np.zeros(nfft)
    H[f <= f1] = 1.0
    m = (f > f1) & (f < f2)
    if rolloff > 0:
        H[m] = np.sqrt(0.5 * (1.0 + np.cos(np.pi * T / rolloff * (f[m] - f1))))
    return H


def shape(symbols, sps, symbol_rate, rolloff):
    """(2, n) symbols -> (2, n*sps) RRC-shaped waveform, unit power per pol."""
    n = symbols.shape[1]
    up = np.zeros((2, n * sps), dtype=complex)
    up[:, ::sps] = symbols
    H = rrc_response(n * sps, symbol_rate * sps, symbol_rate, rolloff)
    out = np.fft.ifft(np.fft.fft(up, axis=1) * H, axis=1)
    p = np.mean(np.abs(out) ** 2, axis=1, keepdims=True)
    return out / np.sqrt(p)


def matched_filter(sig, sps, symbol_rate, rolloff):
    """Apply the receive-side RRC (same response, circular)."""
    H = rrc_response(sig.shape[1], symbol_rate * sps, symbol_rate, rolloff)
    return np.fft.ifft(np.fft.fft(sig, axis=1) * H, axis=1)
