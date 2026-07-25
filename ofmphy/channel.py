"""Baseband equivalent of the optical channel and the electrical front ends.

The fibre carries chromatic dispersion, a polarisation channel (SOP
rotations around a DGD element, plus PDL), combined TX+LO laser phase
noise as a Wiener walk, a TX-LO frequency offset, and ASE loaded as
complex AWGN calibrated through OSNR. By default the fibre is linear,
which is the right regime for a single amplified ZR span at sane launch
powers. Setting nl_steps switches it to a split-step Manakov model with
per-span EDFAs for nonlinear studies.

Both electrical front ends are imperfect as well: the modulator and the
receive hybrid leave a residual I/Q amplitude and quadrature-angle error
plus a differential rail delay (skew), and the ADC quantises. The
receiver owns the matching correction blocks in rxdsp.py.
"""

import numpy as np

C_LIGHT = 299792458.0


def cd_phase(nfft, fs, dispersion_ps_nm_km, length_km, wavelength):
    """Quadratic spectral phase of chromatic dispersion (radians)."""
    D = dispersion_ps_nm_km * 1e-6            # s/m^2
    beta2 = -D * wavelength ** 2 / (2.0 * np.pi * C_LIGHT)
    w = 2.0 * np.pi * np.fft.fftfreq(nfft, 1.0 / fs)
    return 0.5 * beta2 * (length_km * 1e3) * w ** 2


def apply_cd(sig, fs, cfg, invert=False):
    total_km = cfg.fibre_km * cfg.n_spans
    if total_km == 0:
        return sig
    ph = cd_phase(sig.shape[1], fs, cfg.dispersion, total_km, cfg.wavelength)
    rot = np.exp((-1j if invert else 1j) * ph)
    return np.fft.ifft(np.fft.fft(sig, axis=1) * rot, axis=1)


def ssfm_propagate(sig, cfg):
    """Nonlinear fibre, one span: symmetric split-step over the Manakov
    equation,

        dA/dz = -(alpha/2) A - j (beta2/2) d2A/dt2
                + j (8/9) gamma (|Ax|^2 + |Ay|^2) A

    (the 8/9 average holds for fast random birefringence). Each step is a
    half dispersion step, a Kerr rotation using the step's effective length
    L_eff = (1 - e^{-alpha dz})/alpha, attenuation, and the second half
    dispersion step. The lumped EDFA at the span end restores the launch
    power, and the field comes back in the unit-power convention the rest
    of the chain expects. Only the nonlinear phase remembers the Watts.
    """
    n = sig.shape[1]
    fs = cfg.sample_rate
    dz = cfg.fibre_km / cfg.nl_steps
    alpha = np.log(10.0) / 10.0 * cfg.alpha_db_km          # 1/km, power
    leff = (1.0 - np.exp(-alpha * dz)) / alpha if alpha else dz
    half_d = np.exp(0.5j * cd_phase(n, fs, cfg.dispersion, dz, cfg.wavelength))
    p_pol = 10.0 ** ((cfg.launch_power_dbm - 30.0) / 10.0) / 2.0  # W per pol
    scale = np.sqrt(p_pol / np.mean(np.abs(sig) ** 2, axis=1, keepdims=True))
    u = sig * scale
    k_nl = (8.0 / 9.0) * cfg.gamma_w_km
    att = np.exp(-alpha * dz / 2.0)  # field
    for _ in range(cfg.nl_steps):
        u = np.fft.ifft(np.fft.fft(u, axis=1) * half_d, axis=1)
        p = np.abs(u[0]) ** 2 + np.abs(u[1]) ** 2
        u = u * np.exp(1j * k_nl * leff * p) * att
        u = np.fft.ifft(np.fft.fft(u, axis=1) * half_d, axis=1)
    u = u * np.exp(alpha * cfg.fibre_km / 2.0)  # EDFA: undo the span loss
    return u / scale


def osnr_from_launch(launch_dbm, cfg):
    """Amplified-link OSNR budget in the 0.1 nm convention:

        OSNR = P_launch - span_loss + 58 dB - NF - 10 log10(n_spans)

    where -58 dBm is h nu x 12.5 GHz at 193.4 THz and every identical span
    contributes one amplifier's worth of ASE. Feed this to cfg.osnr_db when
    sweeping launch power so ASE and Kerr move together like in a real link.
    """
    span_loss = cfg.alpha_db_km * cfg.fibre_km
    return (launch_dbm - span_loss + 58.0 - cfg.amp_nf_db
            - 10.0 * np.log10(cfg.n_spans))


def apply_polarisation(sig, fs, cfg):
    """R(theta2) . PDL . diag(e^{+j pi f tau}, e^{-j pi f tau}) . R(theta1)."""
    if (cfg.dgd_ps == 0 and cfg.pol_theta1 == 0 and cfg.pol_theta2 == 0
            and cfg.pdl_db == 0):
        return sig
    f = np.fft.fftfreq(sig.shape[1], 1.0 / fs)
    tau = cfg.dgd_ps * 1e-12
    dp = np.exp(1j * np.pi * f * tau)
    c1, s1 = np.cos(cfg.pol_theta1), np.sin(cfg.pol_theta1)
    c2, s2 = np.cos(cfg.pol_theta2), np.sin(cfg.pol_theta2)
    X, Y = np.fft.fft(sig[0]), np.fft.fft(sig[1])
    A = c1 * X + s1 * Y
    Bv = -s1 * X + c1 * Y
    A *= dp
    Bv *= np.conj(dp)
    if cfg.pdl_db:
        # differential loss along the DGD axes, average power preserved
        r = 10.0 ** (cfg.pdl_db / 10.0)
        A *= np.sqrt(2.0 * r / (1.0 + r))
        Bv *= np.sqrt(2.0 / (1.0 + r))
    U = c2 * A + s2 * Bv
    V = -s2 * A + c2 * Bv
    return np.fft.ifft(np.stack([U, V]), axis=1)


def apply_laser(sig, fs, cfg, rng):
    """Common TX+LO phase noise (one Wiener walk) plus frequency offset.

    The frame is processed circularly end to end, so both terms are made
    consistent with the wrap: the walk is pinned into a Brownian bridge
    (identical short-window statistics, no seam) and the offset snaps to
    the FFT grid (within fs/n of the requested value, single-digit MHz for
    typical frames). Otherwise the receive-side CD compensation folds a
    phase discontinuity across the frame boundary and the edge symbols
    take a burst of errors that no real continuous link would see.
    """
    n = sig.shape[1]
    out = sig
    lw = 2.0 * cfg.linewidth_hz  # TX and LO both contribute
    if lw > 0:
        step = np.sqrt(2.0 * np.pi * lw / fs)
        phi = np.cumsum(rng.normal(0.0, step, n))
        phi -= np.arange(n) / n * phi[-1]
        out = out * np.exp(1j * phi)
    if cfg.freq_offset_hz:
        k = round(cfg.freq_offset_hz * n / fs)
        t = np.arange(n) / fs
        out = out * np.exp(2j * np.pi * (k * fs / n) * t)
    return out


def osnr_to_noise_var(osnr_db, symbol_rate, sps, ref_bw=12.5e9, sig_power=1.0):
    """Complex noise variance per sample, per polarisation.

    OSNR is taken in the usual 0.1 nm reference bandwidth over both
    polarisations: OSNR = 2*P_pol / (2*N0*Bref) = P_pol / (N0*Bref),
    so N0 = P_pol / (OSNR * Bref) and sigma^2 = N0 * fs.
    """
    osnr = 10.0 ** (osnr_db / 10.0)
    n0 = sig_power / (osnr * ref_bw)
    return n0 * symbol_rate * sps


def apply_ase(sig, cfg, rng):
    p = float(np.mean(np.abs(sig[0]) ** 2))
    var = osnr_to_noise_var(cfg.osnr_db, cfg.symbol_rate, cfg.sps, cfg.ref_bw, p)
    noise = rng.normal(0.0, np.sqrt(var / 2.0), (2, 2) + sig.shape[1:])
    return sig + noise[0] + 1j * noise[1]


def apply_iq_imperfections(sig, fs, amp_db, phase_deg, skew_ps):
    """Per-polarisation front-end model, same shape at TX and RX.

    The I rail is the reference; the Q rail picks up a gain error g and a
    quadrature-angle error phi: y = I + j g e^{j phi} Q, the usual modulator
    model. Written as y = alpha x + beta x* it leaks a conjugate image at
    20 log10 |beta/alpha| dBc. The rails are also differentially
    delayed by the skew: I by -tau/2, Q by +tau/2, applied as circular
    frequency-domain delays to stay consistent with the frame convention.
    """
    if amp_db == 0 and phase_deg == 0 and skew_ps == 0:
        return sig
    out = np.empty_like(sig)
    g = 10.0 ** (amp_db / 20.0) * np.exp(1j * np.deg2rad(phase_deg))
    if skew_ps:
        w = 2.0 * np.pi * np.fft.fftfreq(sig.shape[1], 1.0 / fs)
        rot = np.exp(1j * w * skew_ps * 1e-12 / 2.0)
    for p in (0, 1):
        i, q = sig[p].real, sig[p].imag
        if skew_ps:
            i = np.fft.ifft(np.fft.fft(i) * rot).real
            q = np.fft.ifft(np.fft.fft(q) * np.conj(rot)).real
        out[p] = i + 1j * g * q
    return out


def adc_quantise(sig, bits, clip_sigma=3.2):
    """Uniform mid-tread quantiser per rail with clipping at clip_sigma rms.
    bits is an ENOB-style knob; 0 means an ideal converter."""
    if bits <= 0:
        return sig
    q = float(2 ** (bits - 1) - 1)
    out = np.empty_like(sig)
    for p in (0, 1):
        rails = []
        for part in (sig[p].real, sig[p].imag):
            fs_scale = clip_sigma * np.sqrt(np.mean(part ** 2))
            rails.append(np.round(np.clip(part / fs_scale, -1, 1) * q) / q * fs_scale)
        out[p] = rails[0] + 1j * rails[1]
    return out


def apply_channel(sig, cfg, rng):
    """Full impairment stack in propagation order: TX front end, fibre
    (linear CD, or split-step Manakov when nl_steps > 0), laser, ASE,
    RX front end, ADC."""
    fs = cfg.sample_rate
    out = apply_iq_imperfections(sig, fs, cfg.tx_iq_amp_db,
                                 cfg.tx_iq_phase_deg, cfg.tx_iq_skew_ps)
    if cfg.nl_steps > 0 and cfg.fibre_km > 0:
        for _ in range(cfg.n_spans):  # EDFA after each span, inside ssfm
            out = ssfm_propagate(out, cfg)
    else:
        out = apply_cd(out, fs, cfg)
    out = apply_polarisation(out, fs, cfg)
    out = apply_laser(out, fs, cfg, rng)
    out = apply_ase(out, cfg, rng)
    out = apply_iq_imperfections(out, fs, cfg.rx_iq_amp_db,
                                 cfg.rx_iq_phase_deg, cfg.rx_iq_skew_ps)
    return adc_quantise(out, cfg.adc_bits, cfg.adc_clip_sigma)


def esn0_db_from_osnr(osnr_db, symbol_rate, ref_bw=12.5e9):
    """Per-polarisation Es/N0 implied by an OSNR (handy for theory curves)."""
    return osnr_db - 10.0 * np.log10(symbol_rate / ref_bw)
