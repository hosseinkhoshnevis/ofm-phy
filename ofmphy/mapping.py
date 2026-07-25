"""Gray-mapped DP-16QAM: bits <-> symbols <-> LLRs.

Bit order per symbol is [XI0 XI1 XQ0 XQ1 YI0 YI1 YQ0 YQ1]; each I/Q rail
is Gray PAM4 with 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3. Symbols are
normalised to unit average energy per polarisation (grid scale 1/sqrt(10)).
"""

import numpy as np

# index (b0<<1)|b1 -> level
GRAY_LEVELS = np.array([-3.0, -1.0, 3.0, 1.0])
_SCALE = 1.0 / np.sqrt(10.0)


def bits_to_symbols(bits):
    """(8n,) bits -> (2, n) complex DP-16QAM symbols, unit power per pol."""
    b = np.asarray(bits, dtype=np.uint8).reshape(-1, 8)
    rails = GRAY_LEVELS[(b[:, 0::2] << 1) | b[:, 1::2]]  # (n, 4): XI XQ YI YQ
    x = rails[:, 0] + 1j * rails[:, 1]
    y = rails[:, 2] + 1j * rails[:, 3]
    return np.stack([x, y]) * _SCALE


def nearest_symbols(z):
    """Slice to the closest 16QAM point (works on any complex array)."""
    u = np.asarray(z) / _SCALE
    qi = 2.0 * np.clip(np.round((u.real + 3.0) / 2.0), 0, 3) - 3.0
    qq = 2.0 * np.clip(np.round((u.imag + 3.0) / 2.0), 0, 3) - 3.0
    return (qi + 1j * qq) * _SCALE


def symbols_to_bits(z):
    """Hard demap (2, n) symbols -> (8n,) bits."""
    u = np.asarray(z) / _SCALE
    out = np.empty((u.shape[1], 8), dtype=np.uint8)
    for pol in (0, 1):
        for comp, off in ((u[pol].real, 0), (u[pol].imag, 2)):
            lev = 2.0 * np.clip(np.round((comp + 3.0) / 2.0), 0, 3) - 3.0
            idx = np.searchsorted([-3.0, -1.0, 1.0, 3.0], lev)  # 0..3 by level
            # invert the Gray map: level order -3,-1,+1,+3 <-> bits 00,01,11,10
            b0 = (idx >= 2).astype(np.uint8)
            b1 = ((idx == 1) | (idx == 2)).astype(np.uint8)
            out[:, 4 * pol + off] = b0
            out[:, 4 * pol + off + 1] = b1
    return out.ravel()


def llrs(z, noise_var):
    """Max-log bit LLRs for (2, n) symbols; positive means bit = 0.

    Max-log is information-lossless for Gray-labelled PAM rails (Ivanov et
    al., IEEE T-IT 2016), so there is nothing to gain from the full
    log-sum-exp here. noise_var is E|n|^2 per complex symbol on the
    unit-energy grid, either a scalar or one value per polarisation. With
    PDL in the link the two pols run at visibly different SNRs, and the
    decoder benefits from knowing that.
    """
    z = np.asarray(z)
    u = np.stack([z[0].real, z[0].imag, z[1].real, z[1].imag]) / _SCALE  # (4, n)
    nv = np.broadcast_to(np.atleast_1d(np.asarray(noise_var, float)), (2,))
    v = np.repeat(np.maximum(nv, 1e-12), 2)[:, None] * 10.0 / 2.0  # per dim
    d = {s: (u - s) ** 2 for s in (-3.0, -1.0, 1.0, 3.0)}
    # b0 splits the rail at zero; b1 separates inner from outer levels
    l0 = (np.minimum(d[1.0], d[3.0]) - np.minimum(d[-1.0], d[-3.0])) / (2.0 * v)
    l1 = (np.minimum(d[-1.0], d[1.0]) - np.minimum(d[-3.0], d[3.0])) / (2.0 * v)
    out = np.stack([l0, l1], axis=-1)          # (4, n, 2)
    return out.transpose(1, 0, 2).reshape(-1)  # (8n,) in tx bit order
