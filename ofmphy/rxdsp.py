"""Receiver DSP chain.

Processing order:

    front-end correction (I/Q deskew + Gram-Schmidt) -> CD compensation ->
    matched filter -> frequency offset estimation -> 2x2 butterfly
    equalizer (CMA, radius-directed, decision-directed) -> blind phase
    search -> frame alignment -> widely-linear TX-IQ correction -> LLRs

The equalizer criteria and the phase search work on the modulus or on
sliced decisions, so they run ahead of carrier recovery. Adaptation uses
block-averaged gradient updates, which reach the same steady state as
per-symbol LMS while each step is one numpy call over a block instead of
a Python-level loop per symbol.

Alignment against the known transmit sequence resolves what a blind
simulation cannot know on its own (polarisation swap, quadrant rotation,
bulk delay) and calibrates the noise variance for the demapper.
Production modems dedicate pilot symbols to the same job; here the
reference sequence is available and plays that role.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .channel import apply_cd
from .mapping import nearest_symbols
from .shaping import matched_filter

R2_CMA = 1.32  # E|s|^4 / E|s|^2 of unit-energy 16QAM
RADII2 = np.array([0.2, 1.0, 1.8])  # the three 16QAM modulus rings, unit energy


def cd_compensate(sig, cfg):
    return apply_cd(sig, cfg.sample_rate, cfg, invert=True)


def frontend_correction(sig, fs, skew_ps):
    """Receive front-end cleanup, first thing in the chain.

    The rail deskew uses the calibrated skew value. Coherent modules
    measure this delay in production and store it in the DSP, and the
    simulation follows the same practice. The I/Q imbalance correction is
    blind: Gram-Schmidt orthogonalisation from the rail statistics, after
    Fatadin and Savory. It needs no carrier lock, since a frequency offset
    spins the constellation but leaves E[I Q] and the rail powers carrying
    the imbalance signature. The transmit-side image is invisible to it
    and is handled by the widely-linear stages after the equalizer.
    """
    out = np.empty_like(sig)
    if skew_ps:
        w = 2.0 * np.pi * np.fft.fftfreq(sig.shape[1], 1.0 / fs)
        rot = np.exp(-1j * w * skew_ps * 1e-12 / 2.0)  # inverse rail delays
    for p in (0, 1):
        i, q = sig[p].real, sig[p].imag
        if skew_ps:
            i = np.fft.ifft(np.fft.fft(i) * rot).real
            q = np.fft.ifft(np.fft.fft(q) * np.conj(rot)).real
        # Gram-Schmidt: I as reference, project it out of Q, renormalise
        pwr = np.mean(i ** 2 + q ** 2)
        i = i / np.sqrt(np.mean(i ** 2))
        q = q - np.mean(i * q) * i
        q = q / np.sqrt(np.mean(q ** 2))
        out[p] = (i + 1j * q) * np.sqrt(pwr / 2.0)
    return out


def wl_correct(sym, ref, n_taps=3, block=4096):
    """Reference-directed widely-linear post filter, per polarisation.

    Transmit-side I/Q imbalance and skew ride through the whole linear
    chain as conjugate mixing (a x + b x*, frequency dependent in the skew
    case), which no strictly linear equalizer can undo. A short
    widely-linear FIR fitted per block cleans it up, and the block-wise
    fitting tracks the slow rotation that the carrier-phase walk puts on
    the image term. This is the batch equivalent of the WL-DD-LMS running
    in real coherent DSPs, with the reference playing the role of the
    decisions.
    """
    if n_taps <= 0:
        return sym
    n = sym.shape[1]
    h = n_taps // 2
    out = np.empty_like(sym)
    eye = 1e-6 * np.eye(2 * n_taps)
    for p in (0, 1):
        pad = np.concatenate([sym[p, -h:], sym[p], sym[p, :h]])
        win = sliding_window_view(pad, n_taps)  # (n, n_taps), centred
        X = np.concatenate([win, np.conj(win)], axis=1)  # (n, 2*n_taps)
        for s in range(0, n, block):
            sel = slice(s, min(s + block, n))
            Xb = X[sel]
            A = Xb.conj().T @ Xb
            b = Xb.conj().T @ ref[p, sel]
            out[p, sel] = X[sel] @ np.linalg.solve(A + eye, b)
    return out


def estimate_freq_offset(sig, fs, search_hz=5e9):
    """4th-power method: 16QAM raised to the 4th leaves a tone at 4*df."""
    n = sig.shape[1]
    spec = np.abs(np.fft.fft(sig ** 4, axis=1)).sum(axis=0)
    f = np.fft.fftfreq(n, 1.0 / fs)
    m = np.abs(f) <= 4.0 * search_hz
    idx = np.flatnonzero(m)[np.argmax(spec[m])]
    # parabolic interpolation around the peak for sub-bin accuracy
    a, b, c = spec[(idx - 1) % n], spec[idx], spec[(idx + 1) % n]
    denom = a - 2 * b + c
    frac = 0.5 * (a - c) / denom if denom else 0.0
    return (f[idx] + frac * (f[1] - f[0])) / 4.0


def remove_freq_offset(sig, fs, f_hz):
    t = np.arange(sig.shape[1]) / fs
    return sig * np.exp(-2j * np.pi * f_hz * t)


class ButterflyEqualizer:
    """T/2-spaced 2x2 MIMO equalizer, CMA then radius-directed.

    With widely_linear=True each output also filters the conjugates of both
    inputs (a 4-branch butterfly). That is the standard answer to transmit
    I/Q imbalance and skew: they ride through every linear stage as
    conjugate mixing, invisible to a strictly linear equalizer. The
    conjugate-branch coupling slowly rotates with the laser phase walk, so
    in WL mode the returned symbols come from the last adaptive sweep
    (tracking, one block behind) rather than a frozen inference pass.
    """

    def __init__(self, n_taps=15, sps=2, block=64, widely_linear=False):
        if n_taps % 2 == 0:
            raise ValueError("odd tap count keeps the delay centred")
        self.n_taps = n_taps
        self.sps = sps
        self.block = block
        self.wl = widely_linear
        h = n_taps // 2
        n_in = 4 if widely_linear else 2
        self.w = np.zeros((2, n_in, n_taps), dtype=complex)  # [out, in, tap]
        self.w[0, 0, h] = 1.0
        self.w[1, 1, h] = 1.0

    def _windows(self, sig):
        h = self.n_taps // 2
        pad = np.concatenate([sig[:, -h:], sig, sig[:, :h]], axis=1)
        win = sliding_window_view(pad, self.n_taps, axis=1)
        win = win[:, ::self.sps]  # (2, n_sym, n_taps), centred per symbol
        if self.wl:
            win = np.concatenate([win, np.conj(win)], axis=0)
        return win

    def _outputs(self, win, sel=slice(None)):
        return np.einsum('opt,pnt->on', self.w, win[:, sel])

    def run(self, sig, cma_sweeps, rde_sweeps, mu_cma, mu_rde):
        """Adapt over the frame; return the equalised symbol stream."""
        sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2, axis=1, keepdims=True))
        win = self._windows(sig)
        self._win = win  # kept for the decision-directed refinement pass
        n_sym = win.shape[1]
        out = np.empty((2, n_sym), dtype=complex)
        plan = [("cma", mu_cma)] * cma_sweeps + [("rde", mu_rde)] * rde_sweeps
        for pass_i, (mode, mu) in enumerate(plan):
            last = pass_i == len(plan) - 1
            for s in range(0, n_sym, self.block):
                sel = slice(s, min(s + self.block, n_sym))
                xb = win[:, sel]
                y = np.einsum('opt,pnt->on', self.w, xb)
                if last:
                    out[:, sel] = y
                r2 = np.abs(y) ** 2
                if mode == "cma":
                    e = R2_CMA - r2
                else:
                    ring = RADII2[np.argmin(np.abs(r2[..., None] - RADII2), axis=-1)]
                    e = ring - r2
                grad = np.einsum('on,pnt->opt', e * y, xb.conj()) / xb.shape[1]
                if self.wl and mode == "cma":
                    # WL-CMA can lock onto the conjugate image during blind
                    # acquisition; keep the conjugate branch frozen until the
                    # radius-directed phase, where it only has to mop up the
                    # small image term.
                    grad[:, 2:] = 0.0
                self.w += mu * grad
            if pass_i == cma_sweeps - 1:
                self._fix_singularity(win)
        if self.wl:
            return out  # tracking outputs: the conjugate branch is time-varying
        return self._outputs(win)

    def refine_dd(self, phase, mu, sweeps):
        """Decision-directed LMS refinement, the usual last adaptation stage.

        CMA and RDE only constrain the modulus, so residual ISI survives
        them. Once blind phase search has produced a carrier track, the
        error can be taken against sliced 16QAM decisions in the
        phase-corrected domain and pushed back through the taps (rotated
        back, since the equalizer itself runs ahead of carrier recovery):

            z = y e^{-j phi},  e = dec(z) - z,  W := W + mu <e e^{+j phi} x*>

        Call after run() and blind_phase_search(); re-estimate the phase on
        the refined output afterwards. Outputs come from the last adaptive
        sweep, same tracking logic as the widely-linear branch.
        """
        win = self._win
        n_sym = win.shape[1]
        out = np.empty((2, n_sym), dtype=complex)
        for sweep in range(sweeps):
            last = sweep == sweeps - 1
            for s in range(0, n_sym, self.block):
                sel = slice(s, min(s + self.block, n_sym))
                xb = win[:, sel]
                y = np.einsum('opt,pnt->on', self.w, xb)
                if last:
                    out[:, sel] = y
                rot = np.exp(-1j * phase[sel])
                z = y * rot
                e = (nearest_symbols(z) - z) * np.conj(rot)
                grad = np.einsum('on,pnt->opt', e, xb.conj()) / xb.shape[1]
                self.w += mu * grad
        return out

    def _fix_singularity(self, win):
        """CMA sometimes converges both outputs onto one polarisation; if so,
        rebuild the second row as the orthogonal filter of the first (linear
        part; the conjugate branch restarts from zero and re-adapts)."""
        y = self._outputs(win, slice(0, 4096))
        c = np.abs(np.vdot(y[0], y[1])) / (np.linalg.norm(y[0]) * np.linalg.norm(y[1]) + 1e-12)
        if c > 0.7:
            self.w[1, 0] = -np.conj(self.w[0, 1][::-1])
            self.w[1, 1] = np.conj(self.w[0, 0][::-1])
            if self.wl:
                self.w[1, 2:] = 0.0


def blind_phase_search(sym, n_phases=32, window=64):
    """Joint-polarisation BPS over the pi/2-symmetric 16QAM constellation.

    Returns derotated symbols and the unwrapped phase track.
    """
    n = sym.shape[1]
    ph = (np.arange(n_phases) / n_phases - 0.5) * (np.pi / 2.0)
    z = sym[:, :, None] * np.exp(-1j * ph)
    d = np.abs(z - nearest_symbols(z)) ** 2
    dist = d.sum(axis=0)  # (n, n_phases): both pols share one laser

    # centred circular moving sum (the frame wraps, so no half windows)
    h = window // 2
    wrapped = np.concatenate([dist[-h:], dist, dist[:h + 1]])
    cs = np.cumsum(np.concatenate([np.zeros((1, n_phases)), wrapped]), axis=0)
    smooth = cs[2 * h + 1:2 * h + 1 + n] - cs[:n]

    raw = ph[np.argmin(smooth, axis=1)]
    # unwrap over the pi/2 ambiguity so the track is continuous
    d_ph = np.diff(raw)
    d_ph = np.mod(d_ph + np.pi / 4.0, np.pi / 2.0) - np.pi / 4.0
    track = raw[0] + np.concatenate([[0.0], np.cumsum(d_ph)])
    return sym * np.exp(-1j * track), track


def ml_phase_refine(sym, window=160, iters=2):
    """Decision-aided ML phase refinement, the second stage of a Zhou-style
    two-stage CPE. BPS quantises the phase to (pi/2)/B and its window is a
    compromise; the ML stage re-estimates the phase continuously from
    sliced decisions,

        h[n] = sum_{|k| <= W/2} y[n+k] dec(y[n+k])*,   phi[n] = arg h[n],

    which removes the test-grid quantisation floor and, at this linewidth-
    symbol-time product (~1e-6), supports a longer window than the blind
    search does. Joint polarisation, circular sums, two passes so the
    second round slices on already-refined symbols.
    """
    y = sym
    n = y.shape[1]
    h = window // 2
    for _ in range(iters):
        prod = (y * np.conj(nearest_symbols(y))).sum(axis=0)  # one LO, both pols
        wrapped = np.concatenate([prod[-h:], prod, prod[:h + 1]])
        cs = np.cumsum(np.concatenate([np.zeros(1, dtype=complex), wrapped]))
        acc = cs[2 * h + 1:2 * h + 1 + n] - cs[:n]
        y = y * np.exp(-1j * np.angle(acc))
    return y


def align_to_reference(sym, ref):
    """Undo bulk delay, polarisation swap and quadrant rotation by circular
    correlation against the known transmit symbols. Returns (aligned,
    noise_var) with aligned in transmit polarisation order."""
    n = sym.shape[1]
    sym = sym / np.sqrt(np.mean(np.abs(sym) ** 2, axis=1, keepdims=True))
    F = np.fft.fft(sym, axis=1)
    G = np.conj(np.fft.fft(ref, axis=1))
    best = {}
    for i in (0, 1):
        for j in (0, 1):
            xc = np.fft.ifft(F[i] * G[j])
            k = int(np.argmax(np.abs(xc)))
            best[i, j] = (float(np.abs(xc[k])), k, xc[k])
    straight = best[0, 0][0] + best[1, 1][0]
    swapped = best[0, 1][0] + best[1, 0][0]
    pairs = [(0, 0), (1, 1)] if straight >= swapped else [(0, 1), (1, 0)]
    out = np.empty_like(ref)
    for i, j in pairs:
        _, k, pk = best[i, j]
        rot = pk / np.abs(pk)
        z = np.roll(sym[i], -k) * np.conj(rot)
        # least-squares complex gain onto the reference grid
        g = np.vdot(z, ref[j]) / np.vdot(z, z)
        out[j] = g * z
    nv = float(np.mean(np.abs(out - ref) ** 2))
    return out, nv
